package com.example.adaptillm.core

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * QueryClassifierML — TFLite-based query classifier.
 *
 * Replaces the rule-based QueryClassifier with a trained neural network.
 * Model: Embedding(32) -> GlobalAvgPool -> Dense(64) -> Dense(4, softmax)
 * Input: INT32 [1, 64] (tokenized + padded)
 * Output: FLOAT32 [1, 4] (softmax probabilities)
 *
 * Latency target: < 5ms per inference on mobile CPU.
 *
 * Falls back to rule-based classification if:
 *   - Model fails to load
 *   - Confidence < CONFIDENCE_THRESHOLD
 */
class QueryClassifierML private constructor(
    private val interpreter: Interpreter,
    private val vocab: Map<String, Int>,
    private val labelMap: Map<Int, String>,
) {
    companion object {
        private const val TAG = "QueryClassifierML"
        private const val MODEL_FILE = "query_classifier.tflite"
        private const val VOCAB_FILE = "vocab.json"
        private const val LABEL_MAP_FILE = "label_map.json"
        private const val MAX_SEQ_LEN = 64
        private const val OOV_INDEX = 1
        private const val CONFIDENCE_THRESHOLD = 0.6f

        @Volatile
        private var instance: QueryClassifierML? = null

        /**
         * Initializes the ML classifier. Call once from Application or Activity.
         * Returns true if model loaded successfully.
         */
        fun init(context: Context): Boolean {
            if (instance != null) return true
            return try {
                val appContext = context.applicationContext
                val model = loadModel(appContext)
                val vocab = loadVocab(appContext)
                val labels = loadLabelMap(appContext)
                val interp = Interpreter(model)
                instance = QueryClassifierML(interp, vocab, labels)
                Log.i(TAG, "ML classifier ready: vocab=${vocab.size} labels=${labels.size}")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Init failed: ${e.message}", e)
                false
            }
        }

        /** Returns true if the ML classifier is available. */
        fun isAvailable(): Boolean = instance != null

        /**
         * Classifies input using ML model with rule-based fallback.
         * This is the drop-in replacement for QueryClassifier.classify().
         */
        fun classify(input: String): QueryType {
            val ml = instance
            if (ml == null) {
                Log.d(TAG, "ML not available, using rule-based")
                return QueryClassifier.classifyRuleBased(input)
            }

            val startNs = System.nanoTime()
            val result = ml.runInference(input)
            val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0
            Log.d(TAG, "'${input.take(30)}' -> ${result.label} (${(result.confidence * 100).toInt()}%) [${String.format("%.1f", elapsedMs)}ms]")

            // Fallback if low confidence
            if (result.confidence < CONFIDENCE_THRESHOLD) {
                val fallback = QueryClassifier.classifyRuleBased(input)
                Log.d(TAG, "Low confidence (${(result.confidence * 100).toInt()}%), fallback: $fallback")
                return fallback
            }

            return result.label
        }

        private fun loadModel(ctx: Context): MappedByteBuffer {
            val fd = ctx.assets.openFd(MODEL_FILE)
            val stream = FileInputStream(fd.fileDescriptor)
            val channel = stream.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }

        private fun loadVocab(ctx: Context): Map<String, Int> {
            val json = ctx.assets.open(VOCAB_FILE).bufferedReader().readText()
            val obj = JSONObject(json)
            val map = HashMap<String, Int>(obj.length())
            for (key in obj.keys()) {
                map[key] = obj.getInt(key)
            }
            return map
        }

        private fun loadLabelMap(ctx: Context): Map<Int, String> {
            val json = ctx.assets.open(LABEL_MAP_FILE).bufferedReader().readText()
            val obj = JSONObject(json)
            val map = HashMap<Int, String>(obj.length())
            for (key in obj.keys()) {
                map[key.toInt()] = obj.getString(key)
            }
            return map
        }
    }

    // Pre-allocated buffers for zero-allocation inference
    private val inputBuffer = ByteBuffer.allocateDirect(MAX_SEQ_LEN * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputArray = Array(1) { FloatArray(labelMap.size) }

    data class ClassificationResult(
        val label: QueryType,
        val confidence: Float,
        val probabilities: FloatArray
    )

    // -----------------------------------------------------------------
    // Tokenizer (matches Python training pipeline exactly)
    // -----------------------------------------------------------------

    private fun cleanText(text: String): String {
        val sb = StringBuilder(text.length)
        for (c in text) {
            when {
                c in 'a'..'z' || c in '0'..'9' -> sb.append(c)
                c in 'A'..'Z' -> sb.append(c + 32)  // lowercase
                c == ' ' || c == '\t' || c == '\n' -> {
                    if (sb.isNotEmpty() && sb.last() != ' ') sb.append(' ')
                }
                else -> {
                    if (sb.isNotEmpty() && sb.last() != ' ') sb.append(' ')
                }
            }
        }
        return sb.toString().trim()
    }

    private fun tokenize(text: String): IntArray {
        val cleaned = cleanText(text)
        val words = cleaned.split(' ').filter { it.isNotEmpty() }
        val result = IntArray(MAX_SEQ_LEN)  // zero-filled = PAD

        val len = minOf(words.size, MAX_SEQ_LEN)
        for (i in 0 until len) {
            result[i] = vocab.getOrDefault(words[i], OOV_INDEX)
        }
        return result
    }

    // -----------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------

    @Synchronized
    fun runInference(input: String): ClassificationResult {
        val tokens = tokenize(input)

        // Fill input buffer
        inputBuffer.rewind()
        for (t in tokens) {
            inputBuffer.putInt(t)
        }

        // Run model
        outputArray[0].fill(0f)
        interpreter.run(inputBuffer, outputArray)

        // Find argmax
        val probs = outputArray[0]
        var maxIdx = 0
        var maxVal = probs[0]
        for (i in 1 until probs.size) {
            if (probs[i] > maxVal) {
                maxVal = probs[i]
                maxIdx = i
            }
        }

        val labelStr = labelMap[maxIdx] ?: "GENERAL"
        val queryType = try { QueryType.valueOf(labelStr) } catch (_: Exception) { QueryType.GENERAL }

        return ClassificationResult(queryType, maxVal, probs.copyOf())
    }
}
