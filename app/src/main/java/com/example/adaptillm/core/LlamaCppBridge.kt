package com.example.adaptillm.core

/**
 * LlamaCppBridge — JNI bridge to the native llama.cpp inference runtime.
 *
 * This object declares the native methods that are implemented in
 * `cpp/llama_interface.cpp`. The native library `llamabridge` must be
 * loaded before any of these methods are called.
 *
 * ## JNI Method Mapping
 *
 * Each `external fun` maps to a C function following JNI naming:
 *   Java_com_example_adaptillm_core_LlamaCppBridge_<methodName>
 *
 * ## Thread Safety
 *
 * The native layer is NOT thread-safe. All calls to [generate] must be
 * serialised by the caller (enforced in [LlamaCppEngine]).
 */
object LlamaCppBridge {

    private var isLoaded = false

    /**
     * Loads the native library. Must be called before any native method.
     *
     * @return true if the library loaded successfully, false otherwise.
     */
    fun loadLibrary(): Boolean {
        if (isLoaded) return true
        return try {
            System.loadLibrary("llamabridge")
            isLoaded = true
            true
        } catch (e: UnsatisfiedLinkError) {
            android.util.Log.e("LlamaCppBridge", "Failed to load native library: ${e.message}", e)
            false
        }
    }

    /**
     * Initialises the llama.cpp model from a GGUF file on disk.
     *
     * @param modelPath Absolute path to the .gguf model file.
     * @return true if the model was loaded successfully.
     */
    external fun nativeInitModel(modelPath: String): Boolean

    /**
     * Runs text generation on the loaded model.
     *
     * @param prompt    The user prompt / instruction text.
     * @param maxTokens Maximum number of tokens to generate.
     * @param threads   Number of CPU threads for inference.
     * @param contextSize  Context window size in tokens.
     * @return Generated text, or an error string prefixed with "ERROR:".
     */
    external fun nativeGenerate(
        prompt: String,
        maxTokens: Int,
        threads: Int,
        contextSize: Int
    ): String

    /**
     * Callback interface for streaming token generation.
     * C++ calls onToken() for each generated token piece.
     */
    interface StreamCallback {
        fun onToken(token: String)
    }

    /**
     * Streaming variant of nativeGenerate.
     * Calls callback.onToken() with each token as it's generated.
     * Returns the full response string when complete.
     */
    external fun nativeGenerateStreaming(
        prompt: String,
        maxTokens: Int,
        threads: Int,
        contextSize: Int,
        callback: StreamCallback
    ): String

    /**
     * Releases all native resources (model, context, KV cache).
     * Safe to call even if no model is loaded.
     */
    external fun nativeFreeModel()

    /**
     * Stops the current generation mid-stream.
     * The native loop checks this flag every token.
     * Thread-safe (uses std::atomic).
     */
    external fun nativeStopGeneration()

    // -----------------------------------------------------------------
    // KV-Cache Reuse API
    // -----------------------------------------------------------------

    /**
     * Pre-evaluates the system prompt and caches the KV state.
     * Call once per mode switch; subsequent generates use the cache.
     *
     * @param sysPrompt  The full system prompt text
     * @param threads    Thread count (must match generate call)
     * @param contextSize Context size (must match generate call)
     * @return true if cache was created successfully
     */
    external fun nativeCacheSystemPrompt(
        sysPrompt: String,
        threads: Int,
        contextSize: Int
    ): Boolean

    /**
     * Generates using cached KV state.
     * Only processes user tokens (system prompt KV is reused).
     * Falls back to full generation if cache is invalid.
     */
    external fun nativeGenerateWithCache(
        userPrompt: String,
        maxTokens: Int,
        threads: Int,
        contextSize: Int,
        callback: StreamCallback
    ): String

    /** Invalidates the KV cache (call on mode switch). */
    external fun nativeInvalidateCache()
}
