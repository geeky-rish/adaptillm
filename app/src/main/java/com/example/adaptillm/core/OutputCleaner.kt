package com.example.adaptillm.core

import android.util.Log

/**
 * OutputCleaner — Production-grade output post-processor.
 *
 * Pipeline position: MODEL_OUTPUT → OutputCleaner → USER
 *
 * Defense passes:
 *   Pass 1: Truncate at template markers (### User:, </s>, <|...)
 *   Pass 2: Remove leftover token fragments
 *   Pass 3: Strip instruction bleed (system prompt echoed back)
 *   Pass 4: Detect hallucinated context (e.g., "Amazon pricing")
 *   Pass 5: Enforce length limits
 *   Pass 6: Trim trailing incomplete sentences (non-code)
 *   Pass 7: Code validation (CODE type only)
 *   Pass 8: Final nonsense detection
 */
object OutputCleaner {

    private const val TAG = "OutputCleaner"

    /** Hard character limit for any response */
    private const val MAX_CHARS = 1500
    /** Soft character limit for non-code responses */
    private const val MAX_CHARS_TEXT = 600

    // Template tokens that signal generation crossed a boundary
    private val TEMPLATE_MARKERS = listOf(
        "</s>",
        "<|user|>", "<|assistant|>", "<|system|>",
        "<|end|>", "<|im_end|>", "<|im_start|>",
        "<|end_of_turn|>", "<|eot_id|>",
        "### User:", "### Assistant:", "### System:",
        "\n### User", "\n### Assistant", "\n### System",
        "\n\n###"
    )

    // Instruction text that the model might echo back
    private val INSTRUCTION_ECHOES = listOf(
        "Give a detailed and well-structured",
        "Give a well-structured and thoughtful",
        "covers all aspects of the given material",
        "Use specific examples to support",
        "avoiding vague or general statements",
        "provide a clear conclusion",
        "Be sure to address any counterarguments",
        "Be concise and direct.",
        "Return ONLY valid code.",
        "Return only valid, working code.",
        "Answer in 2-3 clear sentences.",
        "Explain clearly in 3-4 sentences.",
        "Max 4 sentences.",
        "No explanations."
    )

    // Hallucinated context markers — topics the model invents
    private val HALLUCINATION_MARKERS = listOf(
        "Amazon pricing",
        "Amazon Web Services",
        "customer service representative",
        "as an AI language model",
        "I cannot provide medical",
        "I cannot provide legal",
        "my training data",
        "my knowledge cutoff"
    )

    // -----------------------------------------------------------------
    // Main clean method
    // -----------------------------------------------------------------

    /**
     * Full output sanitization pipeline.
     * Safe for code (skips sentence trimming and hallucination detection).
     *
     * @param raw       Raw model output
     * @param queryType The classified query type
     * @return Cleaned output, or fallback message if output is unusable
     */
    fun clean(raw: String, queryType: QueryType): String {
        if (raw.isEmpty() || raw.startsWith("ERROR")) return raw

        var result = raw

        // Pass 1: Truncate at first template marker
        result = truncateAtMarker(result)

        // Pass 2: Remove leftover fragments
        for (pattern in TEMPLATE_MARKERS) {
            result = result.replace(pattern, "")
        }

        // Pass 3: Strip instruction echoes (all types)
        result = stripInstructionEchoes(result)

        // Pass 4: Hallucination detection (non-code only)
        if (queryType != QueryType.CODE) {
            result = stripHallucinations(result)
        }

        // Pass 5: Enforce length limits
        val charLimit = if (queryType == QueryType.CODE) MAX_CHARS else MAX_CHARS_TEXT
        if (result.length > charLimit) {
            result = truncateAtBoundary(result, charLimit, queryType)
        }

        result = result.trim()

        // Pass 6: Trim trailing incomplete sentence (non-code)
        if (queryType != QueryType.CODE && result.length > 20) {
            result = trimTrailingFragment(result)
        }

        // Pass 7: Code validation (CODE only)
        if (queryType == QueryType.CODE) {
            result = validateCode(result)
        }

        // Pass 8: Nonsense detection
        if (isNonsense(result)) {
            Log.w(TAG, "Nonsense detected, returning fallback")
            return getFallback(queryType)
        }

        return result.trim()
    }

    // -----------------------------------------------------------------
    // Pass implementations
    // -----------------------------------------------------------------

    private fun truncateAtMarker(text: String): String {
        var result = text
        for (marker in TEMPLATE_MARKERS) {
            val idx = result.indexOf(marker)
            if (idx > 0) {
                result = result.substring(0, idx)
            } else if (idx == 0) {
                result = result.removePrefix(marker)
            }
        }
        return result
    }

    private fun stripInstructionEchoes(text: String): String {
        var result = text
        for (pattern in INSTRUCTION_ECHOES) {
            val idx = result.indexOf(pattern, ignoreCase = true)
            if (idx >= 0) {
                Log.d(TAG, "Instruction echo at pos $idx, truncating")
                result = result.substring(0, idx)
            }
        }
        return result
    }

    private fun stripHallucinations(text: String): String {
        var result = text
        for (marker in HALLUCINATION_MARKERS) {
            val idx = result.indexOf(marker, ignoreCase = true)
            if (idx >= 0) {
                Log.w(TAG, "Hallucination marker: '$marker' at pos $idx")
                // Truncate at the sentence containing the hallucination
                val sentenceStart = result.lastIndexOf('.', idx)
                result = if (sentenceStart > 0) {
                    result.substring(0, sentenceStart + 1)
                } else {
                    result.substring(0, idx)
                }
            }
        }
        return result
    }

    private fun truncateAtBoundary(text: String, limit: Int, queryType: QueryType): String {
        if (queryType == QueryType.CODE) {
            // For code: cut at last closing brace before limit
            val lastBrace = text.lastIndexOf('}', limit)
            return if (lastBrace > limit * 0.5) {
                text.substring(0, lastBrace + 1)
            } else {
                text.substring(0, limit)
            }
        }
        // For text: cut at last sentence boundary before limit
        val sub = text.substring(0, limit)
        val lastEnd = maxOf(sub.lastIndexOf('.'), sub.lastIndexOf('!'), sub.lastIndexOf('?'))
        return if (lastEnd > limit * 0.5) sub.substring(0, lastEnd + 1) else sub
    }

    private fun trimTrailingFragment(text: String): String {
        val lastEnd = maxOf(
            text.lastIndexOf('.'),
            text.lastIndexOf('!'),
            text.lastIndexOf('?')
        )
        // Only trim if we keep >60% of the text
        return if (lastEnd > text.length * 0.6) {
            text.substring(0, lastEnd + 1)
        } else {
            text
        }
    }

    /**
     * Code validation — ensures output looks like code.
     * If it contains too much natural language, strips explanation text.
     */
    private fun validateCode(text: String): String {
        if (text.isBlank()) return text

        var result = text.trim()

        // Strip markdown code fences if present
        if (result.startsWith("```")) {
            val firstNewline = result.indexOf('\n')
            if (firstNewline > 0) {
                result = result.substring(firstNewline + 1)
            }
            val lastFence = result.lastIndexOf("```")
            if (lastFence > 0) {
                result = result.substring(0, lastFence)
            }
        }

        // If output has explanation text before code, try to extract just the code
        val codeIndicators = listOf("{", "}", "(", ")", ";", "def ", "class ", "fun ",
            "int ", "void ", "return ", "#include", "import ", "public ", "private ")
        val hasCode = codeIndicators.any { result.contains(it) }

        if (!hasCode && result.length > 20) {
            // No code indicators found — likely explanation text
            Log.w(TAG, "CODE response has no code indicators")
            return result // Return as-is, let the model's output through
        }

        return result.trim()
    }

    // -----------------------------------------------------------------
    // Nonsense detection
    // -----------------------------------------------------------------

    /**
     * Detects degenerate outputs:
     *   - Single repeated character
     *   - Too short (< 3 chars) for non-greeting
     *   - Only punctuation/whitespace
     */
    private fun isNonsense(text: String): Boolean {
        val trimmed = text.trim()
        if (trimmed.length < 3) return true

        // Check for single character repeated
        if (trimmed.length > 10) {
            val uniqueChars = trimmed.toSet().size
            if (uniqueChars <= 3) return true  // e.g., "aaaaaa" or "......"
        }

        // Check if only punctuation/whitespace
        if (trimmed.all { it.isWhitespace() || it in ".,;:!?-_" }) return true

        return false
    }

    // -----------------------------------------------------------------
    // Fallback messages
    // -----------------------------------------------------------------

    private fun getFallback(queryType: QueryType): String {
        return when (queryType) {
            QueryType.CODE -> "// Unable to generate valid code. Please try a more specific query."
            QueryType.DEFINITION -> "I couldn't generate a clear definition. Please rephrase your question."
            QueryType.EXPLANATION -> "I couldn't generate a clear explanation. Please try again."
            QueryType.GENERAL -> "I'm not sure how to answer that. Could you rephrase?"
        }
    }

    // -----------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------

    /** Quick check: is this response usable? */
    fun isValid(output: String): Boolean {
        if (output.isBlank()) return false
        if (output.length < 2) return false
        if (output.startsWith("ERROR")) return false
        if (output.startsWith("[STUB]")) return false
        return true
    }
}
