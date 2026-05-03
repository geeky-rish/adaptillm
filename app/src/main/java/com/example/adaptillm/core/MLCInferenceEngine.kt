package com.example.adaptillm.core

import android.util.Log

/**
 * MLCInferenceEngine — Future MLC LLM backend stub.
 *
 * Placeholder for MLC-LLM Android SDK integration.
 * Implements [LLMInterface] so it can be swapped in without
 * changing any calling code.
 *
 * TODO: Implement using MLC-LLM Android runtime.
 */
class MLCInferenceEngine : LLMInterface {

    companion object {
        private const val TAG = "MLCEngine"
    }

    override fun initModel(): Boolean {
        Log.w(TAG, "MLC engine not yet implemented")
        return false
    }

    override fun generate(input: String, mode: String, callback: (String) -> Unit) {
        callback("MLC engine not yet implemented. Use LlamaCppEngine.")
    }

    override fun generateStreaming(input: String, mode: String, onToken: (String) -> Unit, onComplete: (String) -> Unit) {
        generate(input, mode) { onComplete(it) }
    }

    override fun shutdown() {
        Log.d(TAG, "MLC engine shutdown (no-op)")
    }
}
