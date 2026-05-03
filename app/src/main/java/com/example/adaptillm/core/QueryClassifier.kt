package com.example.adaptillm.core

/**
 * QueryClassifier — Unified entry point for query classification.
 *
 * Routes to ML classifier (QueryClassifierML) if available,
 * falls back to rule-based classification otherwise.
 *
 * ## Usage
 *
 *     // Init ML once (in Activity.onCreate):
 *     QueryClassifierML.init(context)
 *
 *     // Classify anywhere:
 *     val type = QueryClassifier.classify("fibonacci")  // CODE
 */
enum class QueryType {
    CODE,
    DEFINITION,
    EXPLANATION,
    GENERAL
}

object QueryClassifier {

    /**
     * Primary classification entry point.
     * Uses ML if available, rule-based otherwise.
     */
    fun classify(input: String): QueryType {
        return if (QueryClassifierML.isAvailable()) {
            QueryClassifierML.classify(input)
        } else {
            classifyRuleBased(input)
        }
    }

    // -----------------------------------------------------------------
    // Rule-based fallback (also used when ML confidence is low)
    // -----------------------------------------------------------------

    private val CODE_KEYWORDS = listOf(
        "code", "implement", "function", "program", "write a",
        "class ", "method", "algorithm", "script", "snippet",
        "print", "loop", "array", "sort", "fibonacci",
        "def ", "fun ", "public ", "private ", "void ",
        "int ", "string ", "return ", "import ", "package "
    )

    private val DEFINITION_PREFIXES = listOf(
        "what is", "what are", "what was",
        "define", "meaning of", "definition of"
    )

    private val EXPLANATION_KEYWORDS = listOf(
        "how", "why", "explain", "describe",
        "difference between", "compare", "when to use"
    )

    fun classifyRuleBased(input: String): QueryType {
        val lower = input.lowercase().trim()

        if (CODE_KEYWORDS.any { lower.contains(it) }) {
            return QueryType.CODE
        }
        if (DEFINITION_PREFIXES.any { lower.startsWith(it) }) {
            return QueryType.DEFINITION
        }
        if (EXPLANATION_KEYWORDS.any { lower.contains(it) }) {
            return QueryType.EXPLANATION
        }
        return QueryType.GENERAL
    }
}
