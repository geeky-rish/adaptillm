package com.example.adaptillm.core

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.json.JSONArray
import org.json.JSONObject

/**
 * MLQueryClassifier — TFLite-based query classification.
 *
 * Replaces the rule-based [QueryClassifier] with a trained ML model.
 * The model was trained on ~2000 labeled queries and exported to TFLite.
 *
 * ## Architecture
 *
 *     Input text → TextVectorization → Embedding(32) → GlobalAvgPool → Dense(64) → Dense(4)
 *
 * ## Performance
 *
 *     Accuracy: ~93% (vs ~75% rule-based on ambiguous queries)
 *     Latency: <5ms on-device
 *     Model size: ~200KB
 *
 * ## Files Required (place in assets/)
 *
 *     - query_classifier.tflite
 *     - vocab.json
 *     - label_map.json
 *
 * ## Usage
 *
 *     val classifier = MLQueryClassifier.getInstance(context)
 *     val type = classifier.classify("fibonacci")  // → CODE
 *
 * @see QueryClassifier for the fallback rule-based implementation
 */
class MLQueryClassifier private constructor(context: Context) {

    companion object {
        private const val TAG = "MLQueryClassifier"
        private const val MODEL_FILE = "query_classifier.tflite"
        private const val VOCAB_FILE = "vocab.json"
        private const val LABEL_MAP_FILE = "label_map.json"
        private const val MAX_SEQUENCE_LENGTH = 64

        @Volatile
        private var instance: MLQueryClassifier? = null

        /**
         * Thread-safe singleton accessor.
         * Falls back to rule-based classification if TFLite model is unavailable.
         */
        fun getInstance(context: Context): MLQueryClassifier {
            return instance ?: synchronized(this) {
                instance ?: MLQueryClassifier(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }

    private var interpreter: Interpreter? = null
    private var vocabulary: Map<String, Int> = emptyMap()
    private var labelMap: Map<Int, String> = emptyMap()
    private var isModelLoaded = false

    init {
        try {
            loadModel(context)
            loadVocabulary(context)
            loadLabelMap(context)
            isModelLoaded = true
            Log.d(TAG, "ML classifier loaded | vocab=${vocabulary.size} labels=${labelMap.size}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ML classifier, falling back to rules: ${e.message}")
            isModelLoaded = false
        }
    }

    /**
     * Classifies user input into a QueryType.
     *
     * If the ML model is loaded, uses TFLite inference.
     * Otherwise, falls back to rule-based classification.
     *
     * @param input User query text
     * @return QueryType (CODE, DEFINITION, EXPLANATION, GENERAL)
     */
    fun classify(input: String): QueryType {
        if (!isModelLoaded) {
            Log.d(TAG, "ML model not available, using rule-based fallback")
            return QueryClassifier.classify(input)
        }

        return try {
            val startTime = System.nanoTime()
            val result = runInference(input)
            val elapsed = (System.nanoTime() - startTime) / 1_000_000.0
            Log.d(TAG, "classify('${input.take(30)}...') → $result in ${String.format("%.2f", elapsed)}ms")
            result
        } catch (e: Exception) {
            Log.e(TAG, "ML inference failed: ${e.message}, using fallback")
            QueryClassifier.classify(input)
        }
    }

    /**
     * Classifies with confidence score.
     *
     * @return Pair of (QueryType, confidence 0.0-1.0)
     */
    fun classifyWithConfidence(input: String): Pair<QueryType, Float> {
        if (!isModelLoaded) {
            return Pair(QueryClassifier.classify(input), 1.0f)
        }

        return try {
            val scores = runInferenceRaw(input)
            val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: 3
            val confidence = scores[maxIdx]
            val label = labelMap[maxIdx] ?: "GENERAL"
            Pair(QueryType.valueOf(label), confidence)
        } catch (e: Exception) {
            Pair(QueryClassifier.classify(input), 1.0f)
        }
    }

    // -----------------------------------------------------------------
    // Inference internals
    // -----------------------------------------------------------------

    private fun runInference(input: String): QueryType {
        val scores = runInferenceRaw(input)
        val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: 3
        val label = labelMap[maxIdx] ?: "GENERAL"
        return QueryType.valueOf(label)
    }

    private fun runInferenceRaw(input: String): FloatArray {
        val interp = interpreter ?: throw IllegalStateException("Interpreter not loaded")

        // 1. Tokenize: convert text to integer sequence
        val tokenized = tokenize(input)

        // 2. Create input tensor [1, MAX_SEQUENCE_LENGTH]
        val inputArray = Array(1) { tokenized }

        // 3. Create output tensor [1, 4]
        val outputArray = Array(1) { FloatArray(4) }

        // 4. Run inference
        interp.run(inputArray, outputArray)

        return outputArray[0]
    }

    /**
     * Tokenizes text using the loaded vocabulary.
     * Mirrors the TextVectorization layer from the training pipeline.
     *
     * @return IntArray of token IDs, padded to MAX_SEQUENCE_LENGTH
     */
    private fun tokenize(text: String): IntArray {
        val cleaned = text.lowercase()
            .replace(Regex("[^\\w\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()

        val words = cleaned.split(" ")
        val tokenIds = IntArray(MAX_SEQUENCE_LENGTH) { 0 }  // 0 = padding

        for (i in words.indices) {
            if (i >= MAX_SEQUENCE_LENGTH) break
            // vocab index: 0="" (padding), 1="[UNK]", 2+ = actual words
            tokenIds[i] = vocabulary.getOrDefault(words[i], 1)  // 1 = OOV/UNK
        }

        return tokenIds
    }

    // -----------------------------------------------------------------
    // Model loading
    // -----------------------------------------------------------------

    private fun loadModel(context: Context) {
        val modelBuffer = loadAssetFile(context, MODEL_FILE)
        val options = Interpreter.Options()
        options.setNumThreads(2)
        interpreter = Interpreter(modelBuffer, options)
        Log.d(TAG, "TFLite model loaded")
    }

    private fun loadVocabulary(context: Context) {
        val jsonStr = context.assets.open(VOCAB_FILE).bufferedReader().readText()
        val jsonArray = JSONArray(jsonStr)
        val vocabMap = mutableMapOf<String, Int>()
        for (i in 0 until jsonArray.length()) {
            vocabMap[jsonArray.getString(i)] = i
        }
        vocabulary = vocabMap
        Log.d(TAG, "Vocabulary loaded: ${vocabulary.size} tokens")
    }

    private fun loadLabelMap(context: Context) {
        val jsonStr = context.assets.open(LABEL_MAP_FILE).bufferedReader().readText()
        val jsonObj = JSONObject(jsonStr)
        val map = mutableMapOf<Int, String>()
        jsonObj.keys().forEach { key ->
            map[key.toInt()] = jsonObj.getString(key)
        }
        labelMap = map
        Log.d(TAG, "Label map loaded: $labelMap")
    }

    private fun loadAssetFile(context: Context, filename: String): MappedByteBuffer {
        val assetFd = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
    }

    /**
     * Releases TFLite resources.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
        instance = null
        Log.d(TAG, "ML classifier closed")
    }
}
