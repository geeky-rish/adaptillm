package com.example.adaptillm.core

import android.util.Log

/**
 * TFLiteEngine — Future TensorFlow Lite backend stub.
 *
 * Placeholder for TFLite-based inference.
 * Implements [LLMInterface] so it can be swapped in without
 * changing any calling code.
 *
 * TODO: Implement using TFLite Android runtime.
 */
class TFLiteEngine : LLMInterface {

    companion object {
        private const val TAG = "TFLiteEngine"
    }

    override fun initModel(): Boolean {
        Log.w(TAG, "TFLite engine not yet implemented")
        return false
    }

    override fun generate(input: String, mode: String, callback: (String) -> Unit) {
        callback("TFLite engine not yet implemented. Use LlamaCppEngine.")
    }

    override fun generateStreaming(input: String, mode: String, onToken: (String) -> Unit, onComplete: (String) -> Unit) {
        generate(input, mode) { onComplete(it) }
    }

    override fun shutdown() {
        Log.d(TAG, "TFLite engine shutdown (no-op)")
    }
}
