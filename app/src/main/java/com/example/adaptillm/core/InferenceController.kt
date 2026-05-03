package com.example.adaptillm.core

/**
 * InferenceController — Orchestrates adaptive decision making for LLM execution.
 * 
 * Future phases will integrate the real MLC LLM Engine here.
 */
class InferenceController {

    companion object {
        const val MODE_HIGH = "HIGH"
        const val MODE_BALANCED = "BALANCED"
        const val MODE_EFFICIENT = "EFFICIENT"
    }

    /**
     * Internal logic to decide the optimal inference mode based on system state.
     * 
     * Rules:
     * 1. Battery < 20% -> EFFICIENT (Always save power when low)
     * 2. Memory Usage > 70% -> BALANCED (Avoid OOM or heavy swapping)
     * 3. Default -> HIGH (Full quality)
     */
    fun decideMode(battery: Int, memory: Float): String {
        return when {
            battery < 20 -> MODE_EFFICIENT
            memory > 70f -> MODE_BALANCED
            else -> MODE_HIGH
        }
    }
}
