package com.example.adaptillm.core

import android.util.Log

/**
 * ModeTracker — Records mode switches and device state over time.
 *
 * Provides data for the Mode Switch Visualization graph:
 *   X-axis: timestamp
 *   Y-axis: mode (HIGH=3, BALANCED=2, EFFICIENT=1)
 *   Overlay: battery%, memory%
 *
 * ## CSV Export
 *
 *     timestamp, mode, mode_numeric, battery, memory, query_type
 */
class ModeTracker {

    companion object {
        private const val TAG = "ModeTracker"
        const val MODE_HIGH_NUMERIC = 3
        const val MODE_BALANCED_NUMERIC = 2
        const val MODE_EFFICIENT_NUMERIC = 1

        fun modeToNumeric(mode: String): Int = when (mode) {
            InferenceController.MODE_HIGH -> MODE_HIGH_NUMERIC
            InferenceController.MODE_BALANCED -> MODE_BALANCED_NUMERIC
            InferenceController.MODE_EFFICIENT -> MODE_EFFICIENT_NUMERIC
            else -> 0
        }
    }

    data class ModeEvent(
        val timestamp: Long,
        val mode: String,
        val modeNumeric: Int,
        val batteryPercent: Int,
        val memoryPercent: Float,
        val queryType: String,
        val latencyMs: Long = 0
    )

    private val events = mutableListOf<ModeEvent>()

    @Synchronized
    fun record(mode: String, battery: Int, memory: Float, queryType: String, latencyMs: Long = 0) {
        val event = ModeEvent(
            timestamp = System.currentTimeMillis(),
            mode = mode,
            modeNumeric = modeToNumeric(mode),
            batteryPercent = battery,
            memoryPercent = memory,
            queryType = queryType,
            latencyMs = latencyMs
        )
        events.add(event)
        Log.d(TAG, "Mode: $mode | Batt: $battery% | Mem: ${String.format("%.0f", memory)}% | Type: $queryType")
    }

    @Synchronized
    fun getAll(): List<ModeEvent> = events.toList()

    @Synchronized
    fun count(): Int = events.size

    @Synchronized
    fun exportCsv(): String {
        val sb = StringBuilder()
        sb.appendLine("timestamp,mode,mode_numeric,battery,memory,query_type,latency_ms")
        for (e in events) {
            sb.appendLine("${e.timestamp},${e.mode},${e.modeNumeric}," +
                "${e.batteryPercent},${String.format("%.1f", e.memoryPercent)}," +
                "${e.queryType},${e.latencyMs}")
        }
        return sb.toString()
    }

    @Synchronized
    fun clear() {
        events.clear()
        Log.d(TAG, "Mode tracking log cleared")
    }
}
