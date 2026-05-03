package com.example.adaptillm.core

import android.util.Log

/**
 * ConversationManager — Token-aware ring buffer for multi-turn memory.
 *
 * ## Design
 *
 * Unlike naive message-based buffers, this operates on **estimated token count**
 * to ensure conversation history never exceeds the context window. When the
 * budget is exceeded, the oldest turns are pruned first (FIFO).
 *
 * ## Token Estimation
 *
 * Uses a conservative 3.5 chars/token ratio (English average for LLaMA tokenizer).
 * This slightly over-estimates to avoid context overflow.
 *
 * ## Prompt Format (Zephyr/TinyLlama chat template)
 *
 *     <|system|>\n{system_prompt}</s>\n
 *     <|user|>\n{turn_1_user}</s>\n
 *     <|assistant|>\n{turn_1_response}</s>\n
 *     <|user|>\n{turn_N_user}</s>\n
 *     <|assistant|>\n
 *
 * ## Thread Safety
 *
 * All public methods are synchronized. The ConversationManager is owned by
 * LlamaCppEngine and called from a single inference thread.
 */
class ConversationManager {

    companion object {
        private const val TAG = "ConversationManager"
        private const val CHARS_PER_TOKEN = 3.5f
        private const val SYSTEM_PROMPT_RESERVE = 100  // tokens reserved for system prompt
    }

    data class Turn(
        val userMessage: String,
        val assistantResponse: String,
        val estimatedTokens: Int
    )

    private val turns = mutableListOf<Turn>()
    private var maxContextTokens: Int = 512
    private var currentHistoryTokens: Int = 0

    /**
     * Sets the maximum token budget for conversation history.
     * Should be called when context size changes (mode switch).
     *
     * @param contextSize Total context window size in tokens
     * @param maxGenTokens Max tokens reserved for generation
     */
    @Synchronized
    fun setTokenBudget(contextSize: Int, maxGenTokens: Int) {
        // Budget = context - system_prompt - generation_reserve
        maxContextTokens = (contextSize - SYSTEM_PROMPT_RESERVE - maxGenTokens).coerceAtLeast(64)
        Log.d(TAG, "Token budget: $maxContextTokens (ctx=$contextSize, gen=$maxGenTokens)")
        pruneIfNeeded()
    }

    /**
     * Adds a completed turn (user query + assistant response).
     * Automatically prunes oldest turns if budget is exceeded.
     */
    @Synchronized
    fun addTurn(userMessage: String, assistantResponse: String) {
        val tokenEstimate = estimateTokens(userMessage) + estimateTokens(assistantResponse) + 10 // overhead for tags
        val turn = Turn(userMessage, assistantResponse, tokenEstimate)
        turns.add(turn)
        currentHistoryTokens += tokenEstimate
        Log.d(TAG, "Turn added: ${turns.size} turns, ~$currentHistoryTokens tokens")
        pruneIfNeeded()
    }

    /**
     * Builds the conversation history portion of the prompt.
     * Does NOT include the system prompt or the current user query.
     *
     * @return Formatted history string, or empty if no history
     */
    @Synchronized
    fun buildHistoryPrompt(sysToken: String, eosToken: String,
                           usrToken: String, astToken: String): String {
        if (turns.isEmpty()) return ""

        val sb = StringBuilder()
        for (turn in turns) {
            sb.append(usrToken).append("\n")
            sb.append(turn.userMessage).append(eosToken).append("\n")
            sb.append(astToken).append("\n")
            sb.append(turn.assistantResponse).append(eosToken).append("\n")
        }
        return sb.toString()
    }

    /** Returns the number of active turns in memory. */
    @Synchronized
    fun turnCount(): Int = turns.size

    /** Returns estimated token usage of current history. */
    @Synchronized
    fun historyTokens(): Int = currentHistoryTokens

    /** Clears all conversation history. */
    @Synchronized
    fun clear() {
        turns.clear()
        currentHistoryTokens = 0
        Log.d(TAG, "Conversation cleared")
    }

    /** Returns true if there is any conversation history. */
    @Synchronized
    fun hasHistory(): Boolean = turns.isNotEmpty()

    /**
     * Builds clean conversation history using ### format.
     * No Zephyr tokens — prevents template leakage.
     */
    @Synchronized
    fun buildCleanHistory(): String {
        if (turns.isEmpty()) return ""
        val sb = StringBuilder()
        for (turn in turns) {
            sb.append("### User:\n")
            sb.append(turn.userMessage).append("\n\n")
            sb.append("### Assistant:\n")
            sb.append(turn.assistantResponse).append("\n\n")
        }
        return sb.toString()
    }

    // -----------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------

    private fun estimateTokens(text: String): Int {
        return (text.length / CHARS_PER_TOKEN).toInt().coerceAtLeast(1)
    }

    /**
     * Removes oldest turns until history fits within token budget.
     * Preserves the most recent turns (FIFO eviction).
     */
    private fun pruneIfNeeded() {
        var pruned = 0
        while (currentHistoryTokens > maxContextTokens && turns.size > 0) {
            val removed = turns.removeAt(0)
            currentHistoryTokens -= removed.estimatedTokens
            pruned++
        }
        if (pruned > 0) {
            currentHistoryTokens = currentHistoryTokens.coerceAtLeast(0)
            Log.d(TAG, "Pruned $pruned turns, remaining: ${turns.size} (~$currentHistoryTokens tokens)")
        }
    }
}
