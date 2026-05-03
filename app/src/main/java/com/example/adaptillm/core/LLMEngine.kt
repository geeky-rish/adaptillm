package com.example.adaptillm.core

import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * LLMEngine — Mode-aware placeholder for on-device LLM inference.
 *
 * This simulator produces responses whose length and latency vary with the
 * selected inference mode, providing a realistic quality proxy for the
 * adaptive policy engine to optimise against.
 *
 * ## Mode Behaviour
 *
 * | Mode      | Simulated Latency | Response Length | Rationale                    |
 * |-----------|-------------------|-----------------|------------------------------|
 * | HIGH      | ~2 000 ms         | ~500 chars      | Full model, best quality     |
 * | BALANCED  | ~1 200 ms         | ~250 chars      | Quantised / mid-tier model   |
 * | EFFICIENT | ~600 ms           | ~100 chars      | Aggressive quantisation      |
 *
 * A small random jitter (±15 %) is added to both latency and length to
 * simulate real-world variance and prevent the policy engine from
 * over-fitting to deterministic outputs.
 *
 * TODO (Phase 3): Replace this class with [MLCInferenceEngine] backed by
 *  the real MLC Android SDK. The [LLMInterface] contract remains unchanged.
 */
class LLMEngine : LLMInterface {

    override fun initModel(): Boolean = true
    override fun shutdown() {}
    override fun generateStreaming(input: String, mode: String, onToken: (String) -> Unit, onComplete: (String) -> Unit) {
        generate(input, mode) { onComplete(it) }
    }

    companion object {
        private const val TAG = "LLMEngine"

        // Base latency per mode (ms).
        private const val LATENCY_HIGH = 2000L
        private const val LATENCY_BALANCED = 1200L
        private const val LATENCY_EFFICIENT = 600L

        // Base response length per mode (characters).
        private const val LENGTH_HIGH = 500
        private const val LENGTH_BALANCED = 250
        private const val LENGTH_EFFICIENT = 100

        // Jitter range: ±15 %.
        private const val JITTER_FACTOR = 0.15
    }

    /**
     * Simulates asynchronous response generation with mode-dependent
     * latency and output length.
     *
     * Inference runs on a background thread; the [callback] is invoked
     * on the calling thread (typically main) via a [Handler].
     */
    override fun generate(input: String, mode: String, callback: (String) -> Unit) {
        Log.d(TAG, "Inference started | mode=$mode | inputLen=${input.length}")

        Thread {
            try {
                // Determine base parameters for the requested mode.
                val (baseLatency, baseLength) = when (mode) {
                    InferenceController.MODE_HIGH -> Pair(LATENCY_HIGH, LENGTH_HIGH)
                    InferenceController.MODE_BALANCED -> Pair(LATENCY_BALANCED, LENGTH_BALANCED)
                    InferenceController.MODE_EFFICIENT -> Pair(LATENCY_EFFICIENT, LENGTH_EFFICIENT)
                    else -> Pair(LATENCY_BALANCED, LENGTH_BALANCED)
                }

                // Apply jitter.
                val jitteredLatency = applyJitter(baseLatency.toDouble()).toLong().coerceAtLeast(100)
                val jitteredLength = applyJitter(baseLength.toDouble()).toInt().coerceAtLeast(20)

                // Simulate processing time.
                Thread.sleep(jitteredLatency)

                // Generate a response of approximately the target length.
                val response = buildResponse(input, mode, jitteredLength)

                Log.d(TAG, "Inference completed | mode=$mode | latency=${jitteredLatency}ms | " +
                        "responseLen=${response.length}")

                // Deliver result on the main thread.
                Handler(Looper.getMainLooper()).post {
                    callback(response)
                }

            } catch (e: InterruptedException) {
                Log.w(TAG, "Inference interrupted", e)
                Handler(Looper.getMainLooper()).post {
                    callback("Error: Inference was interrupted.")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Inference failed", e)
                Handler(Looper.getMainLooper()).post {
                    callback("Error: ${e.message}")
                }
            }
        }.start()
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /**
     * Builds a synthetic response that is approximately [targetLength] characters
     * long. The content is deterministic but contextualised with the user's input
     * to make the output visually meaningful during development and testing.
     */
    private fun buildResponse(input: String, mode: String, targetLength: Int): String {
        val prefix = "[$mode] "
        val seed = "In response to \"$input\": "
        val filler = "This is a simulated adaptive inference output generated in $mode mode. " +
                "The AdaptiLLM system dynamically selects the inference strategy based on " +
                "real-time device metrics including battery level, memory pressure, and " +
                "historical latency observations. This approach enables Pareto-optimal " +
                "trade-offs between response quality, latency, and energy consumption. "

        val sb = StringBuilder(targetLength + 64)
        sb.append(prefix)
        sb.append(seed)

        // Repeat filler until we reach approximately the target length.
        while (sb.length < targetLength) {
            val remaining = targetLength - sb.length
            if (remaining >= filler.length) {
                sb.append(filler)
            } else {
                sb.append(filler, 0, remaining)
            }
        }

        return sb.toString()
    }

    /**
     * Applies a uniform random jitter of ±[JITTER_FACTOR] to [baseValue].
     */
    private fun applyJitter(baseValue: Double): Double {
        val jitter = 1.0 + (Math.random() * 2 * JITTER_FACTOR - JITTER_FACTOR)
        return baseValue * jitter
    }
}
