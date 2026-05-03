package com.example.adaptillm.core

import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File

/**
 * LlamaCppEngine — Production inference pipeline.
 *
 * Architecture:
 *   INPUT → QueryRouter → [GREETING_BYPASS | LLM_INFERENCE]
 *                                              ↓
 *                                         OutputCleaner
 *                                              ↓
 *                                           OUTPUT
 *
 * Defense layers:
 *   Layer 0: QueryRouter — greetings never touch the model
 *   Layer 1: C++ stop-sequence detection — truncates at ### / <| tokens
 *   Layer 2: OutputCleaner — multi-pass sanitization + validation
 *   Layer 3: UI — replaces streamed text with cleaned final output
 */
class LlamaCppEngine : LLMInterface {

    companion object {
        private const val TAG = "LlamaCppEngine"
        const val MODEL_DIR = "/sdcard/Android/data/com.example.adaptillm/files/llama-model"
        const val ERROR_NO_NATIVE = "Native engine not available. Rebuild with NDK."
        const val ERROR_NO_MODEL = "Model not found. Place a .gguf file in the llama-model directory."
        const val ERROR_INFERENCE = "Model not loaded or native engine error"
    }

    data class InferenceConfig(val maxTokens: Int, val numThreads: Int, val contextSize: Int)

    val conversationManager = ConversationManager()

    // KV-cache tracking
    private var cachedSysPrompt: String? = null
    private var cachedMode: String? = null
    private var cachedCtxSize: Int = 0
    private var cachedThreads: Int = 0

    // Exposed state for metrics logging
    var lastConfig: InferenceConfig = InferenceConfig(128, 3, 256)
        private set
    var lastComplexity: QueryRouter.Complexity = QueryRouter.Complexity.MEDIUM
        private set
    var lastWasGreeting: Boolean = false
        private set
    var lastQueryType: QueryType = QueryType.GENERAL
        private set

    // Experiment mode
    @Volatile var experimentMode: Boolean = false

    private fun modeToConfig(mode: String): InferenceConfig {
        return when (mode) {
            InferenceController.MODE_HIGH -> InferenceConfig(256, 4, 512)
            InferenceController.MODE_BALANCED -> InferenceConfig(128, 3, 256)
            InferenceController.MODE_EFFICIENT -> InferenceConfig(96, 2, 256)
            else -> InferenceConfig(128, 3, 512)
        }
    }

    // State
    private var isModelLoaded = false
    private var isNativeAvailable = false
    @Volatile var isGenerating = false
        private set
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun initModel(): Boolean {
        if (isModelLoaded) return true
        isNativeAvailable = LlamaCppBridge.loadLibrary()
        if (!isNativeAvailable) { Log.e(TAG, "Native library not available"); return false }

        val modelDir = File(MODEL_DIR)
        if (!modelDir.exists() || !modelDir.isDirectory) { Log.e(TAG, "Model dir not found"); return false }

        // Find any .gguf file — supports both TinyLLaMA and upgraded models
        val ggufFile = modelDir.listFiles()?.firstOrNull { it.extension == "gguf" }
        if (ggufFile == null) { Log.e(TAG, "No .gguf file found"); return false }

        Log.d(TAG, "Loading model: ${ggufFile.name} (${ggufFile.length() / 1_000_000}MB)")
        isModelLoaded = try {
            LlamaCppBridge.nativeInitModel(ggufFile.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Model load crash: ${e.message}", e)
            false
        }

        if (isModelLoaded) Log.d(TAG, "Model loaded successfully")
        else Log.e(TAG, "Model init failed")

        return isModelLoaded
    }

    // -----------------------------------------------------------------
    // Batch generation
    // -----------------------------------------------------------------

    override fun generate(input: String, mode: String, callback: (String) -> Unit) {
        val queryType = QueryClassifier.classify(input)
        val baseConfig = modeToConfig(mode)
        val routing = QueryRouter.route(input, queryType, baseConfig)

        lastQueryType = queryType
        lastComplexity = routing.complexity

        // Gate: greeting bypass
        if (routing.route == QueryRouter.Route.GREETING) {
            lastWasGreeting = true
            lastConfig = InferenceConfig(0, 0, 0)
            mainHandler.post { callback(routing.fixedResponse!!) }
            return
        }

        lastWasGreeting = false
        val config = InferenceConfig(routing.maxTokens, routing.numThreads, routing.contextSize)
        lastConfig = config
        conversationManager.setTokenBudget(config.contextSize, config.maxTokens)

        Thread {
            val rawResult = runInference(input, routing.systemPrompt, config)
            val result = OutputCleaner.clean(rawResult, queryType)
            if (OutputCleaner.isValid(result)) {
                conversationManager.addTurn(input, result)
            }
            mainHandler.post { callback(result) }
        }.start()
    }

    // -----------------------------------------------------------------
    // Streaming generation
    // -----------------------------------------------------------------

    override fun generateStreaming(
        input: String,
        mode: String,
        onToken: (String) -> Unit,
        onComplete: (String) -> Unit
    ) {
        val queryType = QueryClassifier.classify(input)
        val baseConfig = modeToConfig(mode)
        val routing = QueryRouter.route(input, queryType, baseConfig)

        lastQueryType = queryType
        lastComplexity = routing.complexity

        // Gate: greeting bypass — deterministic, instant
        if (routing.route == QueryRouter.Route.GREETING) {
            lastWasGreeting = true
            lastConfig = InferenceConfig(0, 0, 0)
            mainHandler.post {
                onToken(routing.fixedResponse!!)
                onComplete(routing.fixedResponse!!)
            }
            return
        }

        lastWasGreeting = false
        val config = InferenceConfig(routing.maxTokens, routing.numThreads, routing.contextSize)
        lastConfig = config
        conversationManager.setTokenBudget(config.contextSize, config.maxTokens)

        Log.d(TAG, "Streaming | mode=$mode route=${routing.route} type=$queryType " +
              "complexity=${routing.complexity} tokens=${config.maxTokens} ctx=${config.contextSize}")

        Thread {
            isGenerating = true
            if (!isModelLoaded) {
                if (!initModel()) {
                    val err = if (!isNativeAvailable) ERROR_NO_NATIVE else ERROR_NO_MODEL
                    mainHandler.post { onComplete(err) }
                    return@Thread
                }
            }

            try {
                val fullPrompt = buildPrompt(input, routing.systemPrompt)
                Log.d(TAG, "Prompt | len=${fullPrompt.length} turns=${conversationManager.turnCount()}")

                val useCache = tryEnsureCache(routing.systemPrompt, mode, config)

                val streamCallback = object : LlamaCppBridge.StreamCallback {
                    override fun onToken(token: String) {
                        if (token.isNotEmpty()) {
                            mainHandler.post { onToken(token) }
                        }
                    }
                }

                val rawResponse: String
                if (useCache && !conversationManager.hasHistory()) {
                    Log.d(TAG, "KV-cache HIT")
                    val userOnly = "### User:\n$input\n\n### Assistant:\n"
                    rawResponse = LlamaCppBridge.nativeGenerateWithCache(
                        userPrompt = userOnly,
                        maxTokens = config.maxTokens,
                        threads = config.numThreads,
                        contextSize = config.contextSize,
                        callback = streamCallback
                    )
                } else {
                    rawResponse = LlamaCppBridge.nativeGenerateStreaming(
                        prompt = fullPrompt,
                        maxTokens = config.maxTokens,
                        threads = config.numThreads,
                        contextSize = config.contextSize,
                        callback = streamCallback
                    )
                }

                // OutputCleaner: full sanitization + validation
                val fullResponse = OutputCleaner.clean(rawResponse, queryType)

                if (OutputCleaner.isValid(fullResponse)) {
                    conversationManager.addTurn(input, fullResponse)
                }

                mainHandler.post { onComplete(fullResponse) }

            } catch (e: Exception) {
                Log.e(TAG, "Streaming error: ${e.message}", e)
                mainHandler.post { onComplete(ERROR_INFERENCE) }
            } finally {
                isGenerating = false
            }
        }.start()
    }

    fun stopGeneration() {
        if (isGenerating && isNativeAvailable) {
            try { LlamaCppBridge.nativeStopGeneration() } catch (_: Exception) {}
            Log.d(TAG, "Stop signal sent")
        }
    }

    // -----------------------------------------------------------------
    // Prompt construction
    // -----------------------------------------------------------------

    private fun buildPrompt(input: String, systemPrompt: String): String {
        val sb = StringBuilder()
        sb.append("### System:\n")
        sb.append(systemPrompt).append("\n\n")

        if (conversationManager.hasHistory()) {
            sb.append(conversationManager.buildCleanHistory())
        }

        sb.append("### User:\n")
        sb.append(input).append("\n\n")
        sb.append("### Assistant:\n")
        return sb.toString()
    }

    // -----------------------------------------------------------------
    // KV-cache
    // -----------------------------------------------------------------

    private fun tryEnsureCache(sysPrompt: String, mode: String, config: InferenceConfig): Boolean {
        val sysText = "### System:\n$sysPrompt\n\n"

        if (cachedSysPrompt == sysText &&
            cachedMode == mode &&
            cachedCtxSize == config.contextSize &&
            cachedThreads == config.numThreads) {
            return true
        }

        Log.d(TAG, "KV-cache MISS: mode $cachedMode -> $mode")
        try { LlamaCppBridge.nativeInvalidateCache() } catch (_: Exception) {}

        val success = try {
            LlamaCppBridge.nativeCacheSystemPrompt(
                sysPrompt = sysText,
                threads = config.numThreads,
                contextSize = config.contextSize
            )
        } catch (e: Exception) {
            Log.e(TAG, "Cache priming failed: ${e.message}")
            false
        }

        if (success) {
            cachedSysPrompt = sysText
            cachedMode = mode
            cachedCtxSize = config.contextSize
            cachedThreads = config.numThreads
        }
        return success
    }

    fun resetConversation() {
        conversationManager.clear()
        cachedSysPrompt = null
        cachedMode = null
        try { LlamaCppBridge.nativeInvalidateCache() } catch (_: Exception) {}
        Log.d(TAG, "Conversation + cache reset")
    }

    // -----------------------------------------------------------------
    // Batch inference
    // -----------------------------------------------------------------

    private fun runInference(input: String, sysPrompt: String, config: InferenceConfig): String {
        if (!isModelLoaded) {
            if (!initModel()) {
                return if (!isNativeAvailable) ERROR_NO_NATIVE else ERROR_NO_MODEL
            }
        }
        return try {
            val prompt = buildPrompt(input, sysPrompt)
            val response = LlamaCppBridge.nativeGenerate(
                prompt = prompt,
                maxTokens = config.maxTokens,
                threads = config.numThreads,
                contextSize = config.contextSize
            )
            if (response.startsWith("ERROR:")) ERROR_INFERENCE else response
        } catch (e: Exception) {
            Log.e(TAG, "Inference exception: ${e.message}", e)
            ERROR_INFERENCE
        }
    }

    override fun shutdown() {
        resetConversation()
        if (isNativeAvailable && isModelLoaded) {
            try { LlamaCppBridge.nativeFreeModel() }
            catch (e: Exception) { Log.e(TAG, "Shutdown error", e) }
        }
        isModelLoaded = false
        Log.d(TAG, "Engine shut down")
    }
}
