package com.example.adaptillm.utils

import android.util.Log

/**
 * Logger — Centralised logging utility for AdaptiLLM.
 *
 * Wraps [android.util.Log] with a consistent tag so every log line
 * can be filtered easily with:
 *     adb logcat -s AdaptiLLM
 *
 * Future enhancements:
 *   - Log level gating (verbose in debug, errors-only in release).
 *   - Optional file-based logging for research data collection.
 */
object Logger {

    private const val TAG = "AdaptiLLM"

    fun d(message: String) {
        Log.d(TAG, message)
    }

    fun i(message: String) {
        Log.i(TAG, message)
    }

    fun w(message: String) {
        Log.w(TAG, message)
    }

    fun e(message: String, throwable: Throwable? = null) {
        Log.e(TAG, message, throwable)
    }
}
