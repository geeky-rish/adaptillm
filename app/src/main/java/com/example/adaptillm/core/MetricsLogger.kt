package com.example.adaptillm.core

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.util.concurrent.atomic.AtomicInteger

/**
 * MetricsLogger — Production-grade telemetry collector.
 *
 * Writes per-inference CSV rows directly to Downloads/adaptillm_logs.csv.
 * Thread-safe, append-only, header written once.
 *
 * Key field: energy_proxy_uams = avg(µA_before, µA_after) × latency_ms
 * This is proportional to energy in joules and has sub-percent resolution,
 * unlike battery_delta which is always 0 for fast inferences.
 */
class MetricsLogger {

    companion object {
        private const val TAG = "MetricsLogger"
        private const val FILE_NAME = "adaptillm_logs.csv"

        private const val CSV_HEADER =
            "query_id,query_text,query_type,mode,complexity,latency_ms,ttft_ms," +
            "battery_start,battery_end,battery_delta," +
            "current_ua_before,current_ua_after,energy_proxy_uams," +
            "tokens_generated,tokens_per_sec,response_length," +
            "context_size,max_tokens,num_threads," +
            "was_truncated,was_greeting,memory_percent,timestamp"
    }

    data class InferenceRecord(
        val queryText: String,
        val queryType: String,
        val mode: String,
        val complexity: String,
        val latencyMs: Long,
        val ttftMs: Long,
        val batteryStart: Int,
        val batteryEnd: Int,
        val batteryDelta: Int,
        val currentUaBefore: Long,
        val currentUaAfter: Long,
        val energyProxyUaMs: Long,
        val tokensGenerated: Int,
        val tokensPerSec: Double,
        val responseLength: Int,
        val contextSize: Int,
        val maxTokens: Int,
        val numThreads: Int,
        val wasTruncated: Boolean,
        val wasGreeting: Boolean,
        val memoryPercent: Float,
        val timestamp: Long
    )

    private val queryCounter = AtomicInteger(0)
    private var csvFile: File? = null
    private val lock = Any()

    /**
     * Initialize with context to resolve Downloads directory.
     * Call once from Activity.onCreate().
     */
    fun init(context: Context) {
        synchronized(lock) {
            if (csvFile != null) return
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (!dir.exists()) dir.mkdirs()
            csvFile = File(dir, FILE_NAME)

            val file = csvFile!!
            if (!file.exists() || file.length() == 0L) {
                try {
                    PrintWriter(FileWriter(file, false)).use { it.println(CSV_HEADER) }
                    Log.i(TAG, "CSV created: ${file.absolutePath}")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to create CSV: ${e.message}")
                }
            } else {
                Log.i(TAG, "CSV exists: ${file.absolutePath} (${file.length()} bytes)")
            }
        }
    }

    /** Logs one inference record. Thread-safe. */
    fun log(record: InferenceRecord): Int {
        val id = queryCounter.incrementAndGet()
        val row = buildRow(id, record)

        synchronized(lock) {
            val file = csvFile
            if (file == null) {
                Log.w(TAG, "CSV not initialized — call init(context) first")
                return id
            }
            try {
                PrintWriter(FileWriter(file, true)).use { it.println(row) }
            } catch (e: Exception) {
                Log.e(TAG, "Write failed: ${e.message}")
            }
        }

        Log.d(TAG, "#$id ${record.mode} | ${record.latencyMs}ms | " +
              "TTFT:${record.ttftMs}ms | ${String.format("%.1f", record.tokensPerSec)} tok/s | " +
              "energy:${record.energyProxyUaMs} uA*ms | greeting:${record.wasGreeting}")
        return id
    }

    /** Returns total queries logged this session. */
    fun count(): Int = queryCounter.get()

    /** Returns path to CSV file, or null. */
    fun csvPath(): String? = csvFile?.absolutePath

    private fun buildRow(id: Int, r: InferenceRecord): String {
        val safeText = r.queryText
            .replace(",", " ")
            .replace("\n", " ")
            .replace("\r", "")
            .take(200)

        return "$id,$safeText,${r.queryType},${r.mode},${r.complexity}," +
               "${r.latencyMs},${r.ttftMs}," +
               "${r.batteryStart},${r.batteryEnd},${r.batteryDelta}," +
               "${r.currentUaBefore},${r.currentUaAfter},${r.energyProxyUaMs}," +
               "${r.tokensGenerated},${String.format("%.2f", r.tokensPerSec)},${r.responseLength}," +
               "${r.contextSize},${r.maxTokens},${r.numThreads}," +
               "${r.wasTruncated},${r.wasGreeting}," +
               "${String.format("%.1f", r.memoryPercent)},${r.timestamp}"
    }
}
