package com.example.adaptillm.core

import android.util.Log
import kotlin.math.exp
import kotlin.math.ln

/**
 * AdaptivePolicyEngine — Multi-objective optimisation module for inference mode selection.
 *
 * This engine replaces simple rule-based switching with a weighted scoring approach
 * that jointly considers latency, energy cost, and quality degradation. Each candidate
 * inference mode is scored and the mode that minimises the composite cost is selected.
 *
 * ## Formulation
 *
 * For each candidate mode **m ∈ {HIGH, BALANCED, EFFICIENT}** the engine computes:
 *
 *     C(m) = w_latency · L̂(m) + w_energy · Ê(m) + w_quality · Q̂(m)
 *
 * where:
 *   - L̂(m): Normalised expected latency for mode m (derived from historical data).
 *   - Ê(m): Normalised energy proxy (function of battery level and estimated drain).
 *   - Q̂(m): Quality penalty = 1 / expected_response_length (shorter ≈ lower quality).
 *
 * The weights are adapted at runtime using softmax over resource pressure signals,
 * ensuring a smooth, differentiable transition between priorities rather than
 * hard-threshold switching.
 *
 * ## Research Relevance
 *
 * This design supports Pareto-front analysis: by varying the weight vector one can
 * trace the latency–energy–quality trade-off surface. The [MetricsLogger] captures
 * per-invocation data to reconstruct this surface offline.
 *
 * @see MetricsLogger
 * @see InferenceController
 */
class AdaptivePolicyEngine {

    companion object {
        private const val TAG = "AdaptivePolicy"

        /** Candidate inference modes ordered by resource intensity (descending). */
        val MODES = arrayOf(
            InferenceController.MODE_HIGH,
            InferenceController.MODE_BALANCED,
            InferenceController.MODE_EFFICIENT
        )
    }

    // ---------------------------------------------------------------
    // State: rolling history for Bayesian-style estimation
    // ---------------------------------------------------------------

    /**
     * Per-mode running statistics. Each entry tracks an exponential moving
     * average (EMA) of observed latency and response length for a given mode.
     */
    private data class ModeStats(
        var emaLatencyMs: Double = 0.0,
        var emaResponseLength: Double = 0.0,
        var observationCount: Int = 0
    )

    private val modeHistory: MutableMap<String, ModeStats> = mutableMapOf(
        InferenceController.MODE_HIGH to ModeStats(emaLatencyMs = 2000.0, emaResponseLength = 500.0),
        InferenceController.MODE_BALANCED to ModeStats(emaLatencyMs = 1200.0, emaResponseLength = 250.0),
        InferenceController.MODE_EFFICIENT to ModeStats(emaLatencyMs = 600.0, emaResponseLength = 100.0)
    )

    /** EMA smoothing factor (α). Higher → faster adaptation, noisier estimates. */
    private val emaSmoothingFactor = 0.3

    // ---------------------------------------------------------------
    // Core: mode selection
    // ---------------------------------------------------------------

    /**
     * Snapshot of device and pipeline state used as input to the policy.
     *
     * @property batteryPercent     Current battery level (0–100).
     * @property memoryUsagePercent Current RAM usage (0–100).
     * @property lastLatencyMs      Latency of the most recent inference call (ms), or -1 if none.
     * @property lastResponseLength Character count of the most recent response, or -1 if none.
     */
    data class SystemSnapshot(
        val batteryPercent: Int,
        val memoryUsagePercent: Float,
        val lastLatencyMs: Long = -1,
        val lastResponseLength: Int = -1
    )

    /**
     * Selects the optimal inference mode by minimising a composite cost function
     * over the candidate set.
     *
     * @param snapshot Current device/pipeline state.
     * @return The mode string that minimises the multi-objective cost.
     */
    fun selectMode(snapshot: SystemSnapshot): String {
        // 1. Compute adaptive weights via softmax over pressure signals.
        val weights = computeAdaptiveWeights(snapshot)

        // 2. Score each candidate mode.
        val scores = MODES.associateWith { mode ->
            computeCost(mode, weights, snapshot)
        }

        Log.d(TAG, "Scores: $scores | Weights: latency=${formatDouble(weights.wLatency)}, " +
                "energy=${formatDouble(weights.wEnergy)}, quality=${formatDouble(weights.wQuality)}")

        // 3. Select the mode with the minimum cost.
        val selectedMode = scores.minByOrNull { it.value }?.key ?: InferenceController.MODE_BALANCED

        Log.d(TAG, "Selected mode: $selectedMode")
        return selectedMode
    }

    // ---------------------------------------------------------------
    // Adaptive weight computation
    // ---------------------------------------------------------------

    /**
     * Container for the three objective weights.
     *
     * Invariant: wLatency + wEnergy + wQuality ≈ 1.0 (softmax normalised).
     */
    data class ObjectiveWeights(
        val wLatency: Double,
        val wEnergy: Double,
        val wQuality: Double
    )

    /**
     * Derives objective weights from current resource pressure using a softmax
     * transformation over three "pressure" signals:
     *
     *   - **Energy pressure**: increases as battery drops (sigmoid-shaped).
     *   - **Latency pressure**: increases when memory usage is high (proxy for
     *     system congestion and potential swap thrashing).
     *   - **Quality pressure**: baseline priority that decreases under resource
     *     pressure (ensures quality is favoured when the device is healthy).
     *
     * The softmax ensures weights are positive, sum to 1, and transition smoothly
     * — unlike hard thresholds which create discontinuous policy jumps.
     */
    private fun computeAdaptiveWeights(snapshot: SystemSnapshot): ObjectiveWeights {
        // Raw pressure signals (higher → more urgent).
        val energyPressure = computeEnergyPressure(snapshot.batteryPercent)
        val latencyPressure = computeLatencyPressure(snapshot.memoryUsagePercent)
        // Quality pressure is the inverse: favour quality when resources are plentiful.
        val qualityPressure = computeQualityPressure(snapshot.batteryPercent, snapshot.memoryUsagePercent)

        // Softmax normalisation.
        val rawScores = doubleArrayOf(latencyPressure, energyPressure, qualityPressure)
        val softmaxWeights = softmax(rawScores)

        return ObjectiveWeights(
            wLatency = softmaxWeights[0],
            wEnergy = softmaxWeights[1],
            wQuality = softmaxWeights[2]
        )
    }

    /**
     * Energy urgency: sigmoid centred at 30% battery.
     * At 100% → ~0.0;  at 30% → 0.5;  at 5% → ~1.0.
     */
    private fun computeEnergyPressure(batteryPercent: Int): Double {
        // Logistic: σ(k · (threshold - x))  with k = 0.15, threshold = 30.
        return sigmoid(0.15 * (30.0 - batteryPercent))
    }

    /**
     * Latency urgency: sigmoid centred at 65% memory usage.
     * Low memory → high pressure to reduce latency (use lighter model).
     */
    private fun computeLatencyPressure(memoryUsagePercent: Float): Double {
        return sigmoid(0.12 * (memoryUsagePercent.toDouble() - 65.0))
    }

    /**
     * Quality preference: high when resources are plentiful, decays under pressure.
     * Combines battery and memory into a single "health" signal.
     */
    private fun computeQualityPressure(batteryPercent: Int, memoryUsagePercent: Float): Double {
        val health = (batteryPercent / 100.0) * (1.0 - memoryUsagePercent / 100.0)
        // Scale to range [0, 1] with a slight bias towards quality when healthy.
        return 0.5 + 0.5 * health
    }

    // ---------------------------------------------------------------
    // Cost computation per mode
    // ---------------------------------------------------------------

    /**
     * Computes the composite cost for a single candidate [mode].
     *
     *     C(m) = w_L · L̂_norm(m) + w_E · Ê(m) + w_Q · Q̂(m)
     *
     * All sub-costs are normalised to [0, 1] for comparability.
     */
    private fun computeCost(mode: String, weights: ObjectiveWeights, snapshot: SystemSnapshot): Double {
        val stats = modeHistory[mode] ?: return Double.MAX_VALUE

        // Normalised latency: map EMA latency to [0, 1] using the max observed.
        val maxLatency = modeHistory.values.maxOfOrNull { it.emaLatencyMs } ?: 1.0
        val normLatency = if (maxLatency > 0) stats.emaLatencyMs / maxLatency else 0.5

        // Energy cost proxy: lighter modes consume less energy.
        // Model as linear mapping: HIGH=1.0, BALANCED=0.6, EFFICIENT=0.25.
        val energyCost = when (mode) {
            InferenceController.MODE_HIGH -> 1.0
            InferenceController.MODE_BALANCED -> 0.6
            InferenceController.MODE_EFFICIENT -> 0.25
            else -> 0.5
        }
        // Modulate by inverse battery: lower battery amplifies energy cost.
        val batteryModulator = 1.0 - (snapshot.batteryPercent / 100.0)
        val normEnergy = energyCost * (0.5 + 0.5 * batteryModulator)

        // Quality penalty: inverse of expected response length (normalised).
        val maxLength = modeHistory.values.maxOfOrNull { it.emaResponseLength } ?: 1.0
        val normQualityPenalty = if (maxLength > 0) {
            1.0 - (stats.emaResponseLength / maxLength)
        } else {
            0.5
        }

        return weights.wLatency * normLatency +
                weights.wEnergy * normEnergy +
                weights.wQuality * normQualityPenalty
    }

    // ---------------------------------------------------------------
    // Feedback: update running statistics after each inference
    // ---------------------------------------------------------------

    /**
     * Updates the EMA statistics for [mode] after an inference completes.
     *
     * This feedback loop allows the policy to adapt to actual model performance
     * rather than relying solely on static priors.
     *
     * @param mode           The inference mode that was used.
     * @param latencyMs      Measured wall-clock latency of the inference call.
     * @param responseLength Character count of the generated response.
     */
    fun recordOutcome(mode: String, latencyMs: Long, responseLength: Int) {
        val stats = modeHistory[mode] ?: return
        val alpha = emaSmoothingFactor

        if (stats.observationCount == 0) {
            // First observation: initialise directly.
            stats.emaLatencyMs = latencyMs.toDouble()
            stats.emaResponseLength = responseLength.toDouble()
        } else {
            stats.emaLatencyMs = alpha * latencyMs + (1 - alpha) * stats.emaLatencyMs
            stats.emaResponseLength = alpha * responseLength + (1 - alpha) * stats.emaResponseLength
        }
        stats.observationCount++

        Log.d(TAG, "Updated stats for $mode: latency=${formatDouble(stats.emaLatencyMs)}ms, " +
                "responseLen=${formatDouble(stats.emaResponseLength)}, n=${stats.observationCount}")
    }

    // ---------------------------------------------------------------
    // Maths utilities
    // ---------------------------------------------------------------

    /** Standard sigmoid: σ(x) = 1 / (1 + e^(-x)). */
    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

    /**
     * Numerically stable softmax over a raw score array.
     * Subtracts the max before exponentiation to avoid overflow.
     */
    private fun softmax(scores: DoubleArray): DoubleArray {
        val maxScore = scores.max()
        val expScores = DoubleArray(scores.size) { exp(scores[it] - maxScore) }
        val sumExp = expScores.sum()
        return DoubleArray(scores.size) { expScores[it] / sumExp }
    }

    /** Format a double to 3 decimal places for logging. */
    private fun formatDouble(value: Double): String = String.format("%.3f", value)
}
