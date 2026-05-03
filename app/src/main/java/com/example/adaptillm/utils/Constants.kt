package com.example.adaptillm.utils

/**
 * Constants — App-wide constant values for AdaptiLLM.
 *
 * Centralises magic numbers and configuration so they can be
 * tuned in one place during future phases.
 */
object Constants {

    /** Application display name. */
    const val APP_NAME = "AdaptiLLM"

    // ----- Device Monitor Thresholds (Phase 2+) -----

    /** Battery percentage below which we consider the device "low". */
    const val LOW_BATTERY_THRESHOLD = 15

    /** Available RAM (MB) below which we consider the device constrained. */
    const val LOW_MEMORY_THRESHOLD_MB = 200L

    /** CPU usage (%) above which we consider the device busy. */
    const val HIGH_CPU_THRESHOLD = 80f

    // ----- Inference Defaults (Phase 2+) -----

    /** Maximum number of tokens to generate per request. */
    const val DEFAULT_MAX_TOKENS = 256

    /** Default temperature for sampling. */
    const val DEFAULT_TEMPERATURE = 0.7f
}
