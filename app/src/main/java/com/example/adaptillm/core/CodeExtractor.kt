package com.example.adaptillm.core

/**
 * CodeExtractor — Extracts code blocks from LLM output.
 *
 * Handles triple-backtick fenced code blocks (``` ... ```).
 * Returns the FIRST complete code block found.
 *
 * Design choice: returns first block rather than concatenating all,
 * because multiple blocks often contain explanatory snippets that
 * shouldn't be mixed into a single paste. Users can copy the full
 * output separately if needed.
 */
object CodeExtractor {

    private val CODE_BLOCK_REGEX = Regex("```(?:\\w*\\n)?([\\s\\S]*?)```")

    /**
     * Extracts the first code block from LLM output.
     *
     * @param output Raw LLM response text.
     * @return Trimmed code content, or empty string if no code block found.
     */
    fun extractCode(output: String): String {
        val match = CODE_BLOCK_REGEX.find(output)
        return match?.groupValues?.get(1)?.trim() ?: ""
    }

    /**
     * Extracts ALL code blocks from LLM output.
     *
     * @param output Raw LLM response text.
     * @return List of trimmed code strings (may be empty).
     */
    fun extractAllCode(output: String): List<String> {
        return CODE_BLOCK_REGEX.findAll(output)
            .map { it.groupValues[1].trim() }
            .filter { it.isNotEmpty() }
            .toList()
    }

    /**
     * Returns true if the output contains at least one code block.
     */
    fun hasCode(output: String): Boolean {
        return CODE_BLOCK_REGEX.containsMatchIn(output)
    }
}
