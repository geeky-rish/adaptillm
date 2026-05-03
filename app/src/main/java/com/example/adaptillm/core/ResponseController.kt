package com.example.adaptillm.core

import android.util.Log

/**
 * ResponseController — Centralized inference control + output sanitization.
 *
 * Responsibilities:
 *   1. Classify input complexity → token/context budget
 *   2. Greeting bypass → deterministic responses without LLM
 *   3. Sanitize model output (multi-pass, safe)
 *   4. Validate output quality
 *
 * Design:
 *   - Greetings get deterministic responses (no LLM call)
 *   - Token budget proportional to complexity
 *   - Multiple defense layers against template leakage
 *   - Safe cleaning that won't corrupt code output
 */
object ResponseController {

    private const val TAG = "ResponseController"

    // ---------------------------------------------------------------
    // Experiment Mode
    // ---------------------------------------------------------------

    /** When true: fixed seed, deterministic sampling, consistent params */
    @Volatile
    var experimentMode: Boolean = false

    // ---------------------------------------------------------------
    // Complexity Classification
    // ---------------------------------------------------------------

    enum class Complexity { TRIVIAL, SHORT, MEDIUM, COMPLEX }

    // Greeting → deterministic response map (no LLM call)
    private val GREETING_RESPONSES = mapOf(
        "hi" to "Hi! How can I help you?",
        "hello" to "Hello! What can I do for you?",
        "hey" to "Hey! How can I help?",
        "yo" to "Hey! What's up?",
        "sup" to "Hey! What can I help with?",
        "ok" to "Got it!",
        "okay" to "Alright!",
        "ok thanks" to "You're welcome!",
        "thanks" to "You're welcome!",
        "thank you" to "You're welcome! Let me know if you need anything else.",
        "bye" to "Goodbye!",
        "goodbye" to "Goodbye! Have a great day!",
        "good morning" to "Good morning!",
        "good evening" to "Good evening!",
        "good night" to "Good night!",
        "sure" to "Great!",
        "yes" to "Got it!",
        "no" to "Alright, let me know if you need anything.",
        "cool" to "Glad to hear!",
        "nice" to "Thanks!",
        "great" to "Glad to help!",
        "awesome" to "Thanks!",
        "fine" to "Good to hear!",
        "good" to "Great!",
        "hmm" to "Take your time! Let me know when you're ready.",
        "wow" to "Glad you think so!"
    )

    /**
     * Check if input is a greeting/trivial input.
     * Returns the fixed response, or null if not a greeting.
     */
    fun getGreetingResponse(input: String): String? {
        val lower = input.lowercase().trim()
        return GREETING_RESPONSES[lower]
    }

    /** Returns true if input is a known greeting/trivial phrase. */
    fun isGreeting(input: String): Boolean {
        return GREETING_RESPONSES.containsKey(input.lowercase().trim())
    }

    /**
     * Classify input complexity based on length, word count, and content.
     * Drives token budget and context size.
     */
    fun classifyComplexity(input: String): Complexity {
        val lower = input.lowercase().trim()

        if (GREETING_RESPONSES.containsKey(lower)) return Complexity.TRIVIAL

        val wordCount = lower.split("\\s+".toRegex()).size

        return when {
            wordCount <= 2 && lower.length < 15 -> Complexity.SHORT
            wordCount <= 6 -> Complexity.MEDIUM
            else -> Complexity.COMPLEX
        }
    }

    // ---------------------------------------------------------------
    // Inference Parameters
    // ---------------------------------------------------------------

    data class InferenceParams(
        val maxTokens: Int,
        val contextSize: Int,
        val numThreads: Int,
        val systemPrompt: String
    )

    private const val SYS_PROMPT_DEFAULT =
        "You are a helpful assistant. Be concise and direct."

    private const val SYS_PROMPT_CODE =
        "You are a coding assistant. Return only valid, working code. No explanations."

    /**
     * Compute inference parameters based on complexity + query type + mode.
     */
    fun getInferenceParams(
        complexity: Complexity,
        queryType: QueryType,
        modeConfig: LlamaCppEngine.InferenceConfig
    ): InferenceParams {

        val sysPrompt = if (queryType == QueryType.CODE) SYS_PROMPT_CODE else SYS_PROMPT_DEFAULT

        val maxTokens = when {
            queryType == QueryType.CODE -> 512
            complexity == Complexity.TRIVIAL -> 24
            complexity == Complexity.SHORT -> 48
            complexity == Complexity.MEDIUM -> modeConfig.maxTokens.coerceAtMost(128)
            else -> modeConfig.maxTokens
        }

        val ctx = when {
            complexity == Complexity.TRIVIAL -> 128
            complexity == Complexity.SHORT -> 256
            queryType == QueryType.CODE -> 768
            else -> modeConfig.contextSize
        }

        val threads = modeConfig.numThreads.coerceAtMost(4)

        Log.d(TAG, "Params: complexity=$complexity type=$queryType -> " +
              "tokens=$maxTokens ctx=$ctx threads=$threads")

        return InferenceParams(maxTokens, ctx, threads, sysPrompt)
    }

    // ---------------------------------------------------------------
    // Output Sanitization (multi-pass)
    // ---------------------------------------------------------------

    private val LEAK_PATTERNS = listOf(
        "</s>",
        "<|user|>", "<|assistant|>", "<|system|>",
        "<|end|>", "<|im_end|>", "<|im_start|>",
        "### User:", "### Assistant:", "### System:",
        "\n### User", "\n### Assistant", "\n### System",
        "\n\n###"
    )

    private val INSTRUCTION_BLEED = listOf(
        "Give a detailed and well-structured",
        "Give a well-structured and thoughtful",
        "covers all aspects of the given material",
        "Use specific examples to support",
        "avoiding vague or general statements",
        "provide a clear conclusion",
        "Be sure to address any counterarguments",
        "Be concise and direct.",
        "Return only valid, working code."
    )

    /**
     * Multi-pass output sanitization.
     *
     * Pass 1: Truncate at first template marker
     * Pass 2: Remove remaining leaked tokens
     * Pass 3: Strip instruction bleed (non-code only)
     * Pass 4: Trim trailing fragments (non-code only)
     */
    fun sanitize(output: String, queryType: QueryType): String {
        if (output.isEmpty()) return output

        var result = output

        // Pass 1: Truncate at first template marker
        for (pattern in LEAK_PATTERNS) {
            val idx = result.indexOf(pattern)
            if (idx > 0) {
                result = result.substring(0, idx)
            } else if (idx == 0) {
                result = result.removePrefix(pattern)
            }
        }

        // Pass 2: Remove leftover fragments
        for (pattern in LEAK_PATTERNS) {
            result = result.replace(pattern, "")
        }

        // Pass 3: Strip instruction bleed (not for code)
        if (queryType != QueryType.CODE) {
            for (pattern in INSTRUCTION_BLEED) {
                val idx = result.indexOf(pattern, ignoreCase = true)
                if (idx >= 0) {
                    result = result.substring(0, idx)
                }
            }
        }

        result = result.trim()

        // Pass 4: Trim trailing incomplete sentence (not for code)
        if (queryType != QueryType.CODE && result.length > 20) {
            val lastEnd = maxOf(
                result.lastIndexOf('.'),
                result.lastIndexOf('!'),
                result.lastIndexOf('?')
            )
            if (lastEnd > result.length * 0.6) {
                result = result.substring(0, lastEnd + 1)
            }
        }

        return result.trim()
    }

    /** Returns true if the response is usable. */
    fun isValidResponse(output: String): Boolean {
        if (output.isBlank()) return false
        if (output.length < 2) return false
        if (output.startsWith("ERROR")) return false
        return true
    }
}
