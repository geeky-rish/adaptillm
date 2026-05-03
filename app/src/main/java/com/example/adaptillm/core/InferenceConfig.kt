package com.example.adaptillm.core

/**
 * InferenceConfig — Model-agnostic configuration for LLM inference.
 *
 * This data class decouples inference parameters from any specific
 * model backend. Both the AdaptivePolicyEngine and individual LLMEngine
 * implementations consume this same config, enabling clean separation.
 *
 * @property maxTokens   Maximum tokens to generate.
 * @property temperature Sampling temperature (0.0 = greedy, 1.0 = creative).
 * @property topK        Top-K sampling parameter.
 * @property topP        Nucleus (top-P) sampling parameter.
 * @property threads     Number of CPU threads for inference.
 * @property contextSize Context window size in tokens.
 */
data class InferenceConfig(
    val maxTokens: Int = 256,
    val temperature: Float = 0.3f,
    val topK: Int = 20,
    val topP: Float = 0.8f,
    val threads: Int = 4,
    val contextSize: Int = 256
)
