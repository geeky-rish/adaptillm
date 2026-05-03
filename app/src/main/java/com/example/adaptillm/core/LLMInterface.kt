package com.example.adaptillm.core

/**
 * LLMInterface — Model-agnostic contract for on-device LLM inference.
 *
 * Supports both batch and streaming generation modes.
 */
interface LLMInterface {

    /** Generates a response asynchronously (batch mode). */
    fun generate(input: String, mode: String, callback: (String) -> Unit)

    /**
     * Generates a response with per-token streaming.
     * [onToken] is called on the main thread with each token piece.
     * [onComplete] is called with the full response when done.
     */
    fun generateStreaming(
        input: String,
        mode: String,
        onToken: (String) -> Unit,
        onComplete: (String) -> Unit
    )

    /** Initialises the model. Idempotent. */
    fun initModel(): Boolean

    /** Releases all backend resources. */
    fun shutdown()
}
