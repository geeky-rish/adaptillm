package com.example.adaptillm.core

import android.util.Log

/**
 * QueryRouter — Deterministic input routing layer.
 *
 * Pipeline: INPUT → QueryRouter → [GREETING | LLM]
 *
 * Responsibilities:
 *   1. Detect greetings → return fixed response (no LLM call)
 *   2. Classify query type → select system prompt + generation params
 *   3. Classify complexity → scale token budget proportionally
 *
 * All routing decisions are deterministic and independent of model state.
 */
object QueryRouter {

    private const val TAG = "QueryRouter"

    // -----------------------------------------------------------------
    // Routing result
    // -----------------------------------------------------------------

    enum class Route { GREETING, DEFINITION, EXPLANATION, CODE, GENERAL }

    data class RoutingResult(
        val route: Route,
        val queryType: QueryType,
        val complexity: Complexity,
        /** Non-null only for GREETING route */
        val fixedResponse: String?,
        /** System prompt for LLM routes */
        val systemPrompt: String,
        /** Generation parameters */
        val maxTokens: Int,
        val contextSize: Int,
        val numThreads: Int
    )

    // -----------------------------------------------------------------
    // Complexity
    // -----------------------------------------------------------------

    enum class Complexity { TRIVIAL, SHORT, MEDIUM, COMPLEX }

    // -----------------------------------------------------------------
    // Greeting map — deterministic, zero-latency responses
    // -----------------------------------------------------------------

    private val GREETINGS = mapOf(
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

    // -----------------------------------------------------------------
    // System prompts — minimal to prevent instruction bleed
    // -----------------------------------------------------------------

    private const val SYS_DEFINITION = "Answer in 2-3 clear sentences. No examples."
    private const val SYS_EXPLANATION = "Explain clearly in 3-4 sentences."
    private const val SYS_CODE = "Return ONLY valid code. No explanations."
    private const val SYS_GENERAL = "Be concise. Max 4 sentences."

    // -----------------------------------------------------------------
    // Generation parameter table
    // -----------------------------------------------------------------
    //
    //  Type        | temp | top_p | max_tokens | ctx  | repeat_penalty
    //  ------------|------|-------|------------|------|---------------
    //  DEFINITION  | 0.15 | 0.85  | 96         | 256  | 1.3
    //  EXPLANATION | 0.20 | 0.85  | 160        | 384  | 1.3
    //  CODE        | 0.10 | 0.90  | 512        | 768  | 1.2
    //  GENERAL     | 0.20 | 0.85  | 128        | 256  | 1.3
    //
    // These are passed via context; sampler params are set in C++.
    // Token budgets and context are controlled here in Kotlin.

    // -----------------------------------------------------------------
    // Route
    // -----------------------------------------------------------------

    /**
     * Route the input. Returns a complete RoutingResult.
     *
     * @param input     Raw user input
     * @param queryType Pre-classified query type (from QueryClassifier)
     * @param modeConfig Base config from adaptive mode selection
     */
    fun route(
        input: String,
        queryType: QueryType,
        modeConfig: LlamaCppEngine.InferenceConfig
    ): RoutingResult {
        val lower = input.lowercase().trim()

        // Gate 1: Greeting bypass
        val greeting = GREETINGS[lower]
        if (greeting != null) {
            Log.d(TAG, "GREETING: '$lower'")
            return RoutingResult(
                route = Route.GREETING,
                queryType = queryType,
                complexity = Complexity.TRIVIAL,
                fixedResponse = greeting,
                systemPrompt = "",
                maxTokens = 0,
                contextSize = 0,
                numThreads = 0
            )
        }

        // Gate 2: Classify complexity
        val wordCount = lower.split("\\s+".toRegex()).size
        val complexity = when {
            wordCount <= 2 && lower.length < 15 -> Complexity.SHORT
            wordCount <= 6 -> Complexity.MEDIUM
            else -> Complexity.COMPLEX
        }

        // Gate 3: Map query type → route + params
        val (route, sysPrompt, baseTokens, baseCtx) = when (queryType) {
            QueryType.CODE -> Quad(Route.CODE, SYS_CODE, 512, 768)
            QueryType.DEFINITION -> Quad(Route.DEFINITION, SYS_DEFINITION, 96, 256)
            QueryType.EXPLANATION -> Quad(Route.EXPLANATION, SYS_EXPLANATION, 160, 384)
            QueryType.GENERAL -> Quad(Route.GENERAL, SYS_GENERAL, 128, 256)
        }

        // Gate 4: Scale tokens by complexity
        val maxTokens = when (complexity) {
            Complexity.TRIVIAL -> 24
            Complexity.SHORT -> baseTokens.coerceAtMost(48)
            Complexity.MEDIUM -> baseTokens.coerceAtMost(modeConfig.maxTokens)
            Complexity.COMPLEX -> baseTokens
        }

        val ctx = when (complexity) {
            Complexity.TRIVIAL -> 128
            Complexity.SHORT -> 256
            else -> baseCtx.coerceAtLeast(modeConfig.contextSize)
        }

        val threads = modeConfig.numThreads.coerceAtMost(4)

        Log.d(TAG, "$route | type=$queryType complexity=$complexity tokens=$maxTokens ctx=$ctx")

        return RoutingResult(
            route = route,
            queryType = queryType,
            complexity = complexity,
            fixedResponse = null,
            systemPrompt = sysPrompt,
            maxTokens = maxTokens,
            contextSize = ctx,
            numThreads = threads
        )
    }

    /** Returns true if input is a known greeting. */
    fun isGreeting(input: String): Boolean = GREETINGS.containsKey(input.lowercase().trim())

    // Helper data class for destructuring
    private data class Quad<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)
}
