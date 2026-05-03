package com.example.adaptillm.core

import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * BenchmarkRunner — Compares adaptive vs static inference.
 *
 * Runs the same query batch through both modes and collects
 * latency, tok/s, energy, and response metrics for analysis.
 *
 * ## Usage
 *
 *     val runner = BenchmarkRunner(engine, deviceMonitor, policyEngine, metricsLogger)
 *     runner.run(queries) { results ->
 *         val csv = BenchmarkRunner.exportCsv(results)
 *         // save or display csv
 *     }
 *
 * ## CSV Columns
 *
 *     query_id, mode_type, mode_selected, query_type, latency_ms, ttft_ms,
 *     tokens, response_length, battery_before, battery_after, battery_delta
 */
class BenchmarkRunner(
    private val engine: LLMInterface,
    private val deviceMonitor: DeviceMonitor,
    private val policyEngine: AdaptivePolicyEngine,
    private val metricsLogger: MetricsLogger
) {
    companion object {
        private const val TAG = "BenchmarkRunner"

        // Static baseline config (fixed, no adaptation)
        const val STATIC_TOKENS = 256
        const val STATIC_THREADS = 4
        const val STATIC_CTX = 512

        /**
         * Standard benchmark query set covering all query types.
         * Includes ambiguous queries to test classifier accuracy.
         */
        val BENCHMARK_QUERIES = listOf(
            "write a python function to sort an array",          // CODE
            "fibonacci",                                          // CODE (ambiguous)
            "implement binary search in java",                    // CODE
            "what is machine learning",                           // DEFINITION
            "define recursion",                                   // DEFINITION
            "how does TCP handshake work",                        // EXPLANATION
            "explain the difference between stack and heap",      // EXPLANATION
            "hello",                                              // GENERAL
            "suggest a good programming language",                // GENERAL
            "compare REST and GraphQL",                           // EXPLANATION
        )

        fun exportCsv(results: List<BenchmarkResult>): String {
            val sb = StringBuilder()
            sb.appendLine("query_id,mode_type,mode_selected,query_type," +
                "latency_ms,tokens,response_length,battery_before," +
                "battery_after,battery_delta")
            for (r in results) {
                sb.appendLine("${r.queryId},${r.modeType},${r.modeSelected}," +
                    "${r.queryType},${r.latencyMs},${r.tokenCount}," +
                    "${r.responseLength},${r.batteryBefore}," +
                    "${r.batteryAfter},${r.batteryDelta}")
            }
            return sb.toString()
        }
    }

    data class BenchmarkResult(
        val queryId: Int,
        val query: String,
        val modeType: String,       // "adaptive" or "static"
        val modeSelected: String,   // HIGH/BALANCED/EFFICIENT or STATIC
        val queryType: String,
        val latencyMs: Long,
        val tokenCount: Int,
        val responseLength: Int,
        val batteryBefore: Int,
        val batteryAfter: Int,
        val batteryDelta: Int
    )

    /**
     * Runs the full benchmark: all queries in adaptive mode, then static mode.
     * Results are delivered via callback on the main thread.
     */
    fun run(
        context: android.content.Context,
        queries: List<String> = BENCHMARK_QUERIES,
        onProgress: (current: Int, total: Int) -> Unit = { _, _ -> },
        onComplete: (List<BenchmarkResult>) -> Unit
    ) {
        val mainHandler = Handler(Looper.getMainLooper())
        val results = mutableListOf<BenchmarkResult>()
        val totalRuns = queries.size * 2  // adaptive + static

        Thread {
            var runIndex = 0

            // --- Pass 1: Adaptive mode ---
            for ((idx, query) in queries.withIndex()) {
                val queryType = QueryClassifier.classify(query)
                val battBefore = deviceMonitor.getBatteryLevel(context)
                val memory = deviceMonitor.getMemoryUsage(context)

                val snapshot = AdaptivePolicyEngine.SystemSnapshot(
                    batteryPercent = battBefore,
                    memoryUsagePercent = memory,
                    lastLatencyMs = -1,
                    lastResponseLength = -1
                )
                val mode = policyEngine.selectMode(snapshot)

                val startMs = System.currentTimeMillis()
                var response = ""
                var tokenCount = 0

                // Synchronous generation
                val latch = java.util.concurrent.CountDownLatch(1)
                engine.generateStreaming(query, mode,
                    onToken = { tokenCount++ },
                    onComplete = { resp -> response = resp; latch.countDown() }
                )
                latch.await()

                val latencyMs = System.currentTimeMillis() - startMs
                val battAfter = deviceMonitor.getBatteryLevel(context)

                results.add(BenchmarkResult(
                    queryId = idx, query = query,
                    modeType = "adaptive", modeSelected = mode,
                    queryType = queryType.name, latencyMs = latencyMs,
                    tokenCount = tokenCount, responseLength = response.length,
                    batteryBefore = battBefore, batteryAfter = battAfter,
                    batteryDelta = (battBefore - battAfter).coerceAtLeast(0)
                ))

                runIndex++
                val progress = runIndex
                mainHandler.post { onProgress(progress, totalRuns) }
                Log.d(TAG, "Adaptive[$idx]: ${query.take(30)} → ${mode} ${latencyMs}ms")
            }

            // --- Pass 2: Static mode ---
            for ((idx, query) in queries.withIndex()) {
                val queryType = QueryClassifier.classify(query)
                val battBefore = deviceMonitor.getBatteryLevel(context)

                val startMs = System.currentTimeMillis()
                var response = ""
                var tokenCount = 0

                val latch = java.util.concurrent.CountDownLatch(1)
                engine.generateStreaming(query, "STATIC",
                    onToken = { tokenCount++ },
                    onComplete = { resp -> response = resp; latch.countDown() }
                )
                latch.await()

                val latencyMs = System.currentTimeMillis() - startMs
                val battAfter = deviceMonitor.getBatteryLevel(context)

                results.add(BenchmarkResult(
                    queryId = idx, query = query,
                    modeType = "static", modeSelected = "STATIC",
                    queryType = queryType.name, latencyMs = latencyMs,
                    tokenCount = tokenCount, responseLength = response.length,
                    batteryBefore = battBefore, batteryAfter = battAfter,
                    batteryDelta = (battBefore - battAfter).coerceAtLeast(0)
                ))

                runIndex++
                val progress = runIndex
                mainHandler.post { onProgress(progress, totalRuns) }
                Log.d(TAG, "Static[$idx]: ${query.take(30)} → STATIC ${latencyMs}ms")
            }

            mainHandler.post { onComplete(results) }
            Log.d(TAG, "Benchmark complete: ${results.size} results")
        }.start()
    }
}
