/**
 * llama_interface.cpp — Optimized JNI bridge to llama.cpp for AdaptiLLM.
 *
 * KEY OPTIMIZATION: Context is created ONCE and reused across queries.
 * Only the KV-cache is cleared between queries (via llama_kv_cache_clear),
 * NOT the entire context. This eliminates the ~2-5 second context creation
 * overhead that was causing high TTFT.
 *
 * STOP SUPPORT: g_stop_requested flag allows Kotlin to abort generation
 * mid-stream via nativeStopGeneration().
 */

#include <jni.h>
#include <string>
#include <vector>
#include <cstring>
#include <atomic>
#include <android/log.h>

#ifdef LLAMA_AVAILABLE
#include "llama.h"
#endif

#define LOG_TAG "LlamaCppNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

#ifdef LLAMA_AVAILABLE
static llama_model   *g_model = nullptr;
static llama_context *g_ctx   = nullptr;
#endif

static bool g_model_loaded    = false;
static bool g_backend_inited  = false;

// Persistent context parameters (to detect when rebuild is needed)
static int32_t g_ctx_size    = 0;
static int32_t g_ctx_threads = 0;

// Stop flag: set from Kotlin to abort generation
static std::atomic<bool> g_stop_requested(false);

// ---------------------------------------------------------------------------
// Stop-sequence detection: prevents template token leakage at the source
// ---------------------------------------------------------------------------

static const char* STOP_SEQUENCES[] = {
    "### User:",
    "### System:",
    "### Assistant:",
    "\n\n###",
    "\n###",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "</s>",
    "<|end",
    "<|im_end",
    nullptr  // sentinel
};

/**
 * Checks if the generated text contains any stop sequence.
 * If found, truncates the result to just before the stop sequence.
 * Returns true if a stop sequence was detected.
 */
static bool check_stop_sequences(std::string &result) {
    for (int i = 0; STOP_SEQUENCES[i] != nullptr; i++) {
        size_t pos = result.find(STOP_SEQUENCES[i]);
        if (pos != std::string::npos) {
            result.resize(pos);
            LOGD("Stop sequence '%s' detected at pos %zu, truncated", STOP_SEQUENCES[i], pos);
            return true;
        }
    }
    return false;
}

/**
 * Check if the tail of the result might be the beginning of a stop sequence.
 * Used to hold back streaming tokens that could be part of a stop sequence.
 * Returns the number of suspicious trailing characters (0 = safe to stream).
 */
static int check_partial_stop(const std::string &result) {
    if (result.size() < 2) return 0;
    // Check if result ends with characters that could start a stop sequence
    size_t len = result.size();
    // Look at last 12 chars max (longest stop sequence is ~16 chars)
    int check_len = (len < 12) ? (int)len : 12;
    for (int tail = 1; tail <= check_len; tail++) {
        std::string suffix = result.substr(len - tail);
        for (int i = 0; STOP_SEQUENCES[i] != nullptr; i++) {
            std::string seq(STOP_SEQUENCES[i]);
            if (seq.substr(0, tail) == suffix) {
                return tail;  // trailing chars match start of stop sequence
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Helper: ensure context exists with given params, reuse if possible
// ---------------------------------------------------------------------------
#ifdef LLAMA_AVAILABLE
static bool ensure_context(int32_t ctx_size, int32_t threads) {
    // Reuse existing context if params match
    if (g_ctx != nullptr && g_ctx_size == ctx_size && g_ctx_threads == threads) {
        return true;  // context already valid
    }

    // Must rebuild context (params changed)
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
        LOGI("Context freed (params changed: ctx %d->%d, threads %d->%d)",
             g_ctx_size, ctx_size, g_ctx_threads, threads);
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = (uint32_t)ctx_size;
    ctx_params.n_batch         = (uint32_t)ctx_size;
    ctx_params.n_threads       = threads;
    ctx_params.n_threads_batch = threads;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (g_ctx == nullptr) {
        LOGE("llama_init_from_model failed (ctx=%d, threads=%d)", ctx_size, threads);
        return false;
    }

    g_ctx_size = ctx_size;
    g_ctx_threads = threads;
    LOGI("Context created: ctx=%d, threads=%d", ctx_size, threads);
    return true;
}
#endif

// ---------------------------------------------------------------------------
// JNI: nativeInitModel
// ---------------------------------------------------------------------------

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_adaptillm_core_LlamaCppBridge_nativeInitModel(
        JNIEnv *env,
        jobject /* this */,
        jstring modelPath) {

    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) {
        LOGE("nativeInitModel: null modelPath");
        return JNI_FALSE;
    }

    LOGI("nativeInitModel: path=%s", path);

#ifdef LLAMA_AVAILABLE

    // 1. Initialize backend (once)
    if (!g_backend_inited) {
        llama_backend_init();
        g_backend_inited = true;
        LOGD("llama_backend_init() done");
    }

    // 2. Free previous model if any
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_model_loaded = false;
    g_ctx_size = 0;
    g_ctx_threads = 0;

    // 3. Load model
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU-only
    model_params.use_mmap = true;

    g_model = llama_model_load_from_file(path, model_params);
    if (g_model == nullptr) {
        LOGE("llama_model_load_from_file failed: %s", path);
        env->ReleaseStringUTFChars(modelPath, path);
        return JNI_FALSE;
    }
    LOGI("Model loaded successfully");

    // 4. Create initial context (will be reused!)
    if (!ensure_context(512, 4)) {
        llama_model_free(g_model);
        g_model = nullptr;
        env->ReleaseStringUTFChars(modelPath, path);
        return JNI_FALSE;
    }

    g_model_loaded = true;

#else
    g_model_loaded = true;
    LOGW("STUB MODE: model 'loaded' (llama.cpp not linked)");
#endif

    env->ReleaseStringUTFChars(modelPath, path);
    return JNI_TRUE;
}

// ---------------------------------------------------------------------------
// JNI: nativeGenerate (batch, non-streaming)
// ---------------------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_adaptillm_core_LlamaCppBridge_nativeGenerate(
        JNIEnv *env,
        jobject /* this */,
        jstring prompt,
        jint maxTokens,
        jint threads,
        jint contextSize) {

    if (!g_model_loaded) {
        return env->NewStringUTF("ERROR: Model not loaded");
    }

    const char *prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
    if (prompt_cstr == nullptr) {
        return env->NewStringUTF("ERROR: null prompt");
    }

    LOGI("nativeGenerate: prompt_len=%zu, max_tokens=%d, threads=%d, ctx=%d",
         strlen(prompt_cstr), (int)maxTokens, (int)threads, (int)contextSize);

#ifdef LLAMA_AVAILABLE

    // Reuse context if possible, rebuild only if params changed
    if (!ensure_context(contextSize, threads)) {
        env->ReleaseStringUTFChars(prompt, prompt_cstr);
        return env->NewStringUTF("ERROR: Failed to create inference context");
    }

    // Clear KV-cache (NOT the context!) - this is the key optimization
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Tokenize
    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);

    int32_t n_prompt_tokens = llama_tokenize(
        vocab, prompt_cstr, (int32_t)strlen(prompt_cstr),
        nullptr, 0, true, false
    );
    if (n_prompt_tokens < 0) n_prompt_tokens = -n_prompt_tokens;

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    int32_t actual = llama_tokenize(
        vocab, prompt_cstr, (int32_t)strlen(prompt_cstr),
        prompt_tokens.data(), (int32_t)prompt_tokens.size(),
        true, false
    );

    if (actual < 0) {
        LOGE("Tokenization failed, returned %d", actual);
        env->ReleaseStringUTFChars(prompt, prompt_cstr);
        return env->NewStringUTF("ERROR: Tokenization failed");
    }
    prompt_tokens.resize(actual);

    // Truncate if prompt exceeds context
    if (actual >= contextSize) {
        LOGW("Prompt (%d tokens) exceeds context (%d), truncating", actual, contextSize);
        prompt_tokens.resize(contextSize - 1);
    }

    env->ReleaseStringUTFChars(prompt, prompt_cstr);

    LOGD("Tokenized: %d tokens", actual);

    // Prefill
    struct llama_batch batch = llama_batch_get_one(
        prompt_tokens.data(), (int32_t)prompt_tokens.size()
    );
    int32_t decode_result = llama_decode(g_ctx, batch);
    if (decode_result != 0) {
        LOGE("Prefill decode failed: %d", decode_result);
        return env->NewStringUTF("ERROR: Prefill decode failed");
    }

    // Sampler — tight settings for focused generation
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.3f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(15));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.85f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.2f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    // Generate
    std::string result;
    result.reserve(maxTokens * 8);
    int softLimit = (int)(maxTokens * 0.80);

    for (int i = 0; i < maxTokens; i++) {
        llama_token new_token = llama_sampler_sample(smpl, g_ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            LOGD("EOG at step %d", i);
            break;
        }

        llama_sampler_accept(smpl, new_token);

        char piece[256];
        int32_t n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, false);
        if (n_chars > 0) {
            result.append(piece, n_chars);
        }

        // Stop-sequence detection: prevents template leakage
        if (check_stop_sequences(result)) {
            LOGD("Stop-sequence break at step %d", i);
            break;
        }

        if (i >= softLimit && !result.empty()) {
            char lastChar = result.back();
            if (lastChar == '.' || lastChar == '!' || lastChar == '?' || lastChar == '\n') {
                LOGD("Safe stop at step %d/%d", i, maxTokens);
                break;
            }
        }

        struct llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("Decode failed at step %d", i);
            break;
        }
    }

    llama_sampler_free(smpl);
    LOGI("Generation complete: %zu chars", result.size());
    return env->NewStringUTF(result.c_str());

#else
    std::string stub = "[NATIVE STUB] prompt_len=";
    stub += std::to_string((int)strlen(prompt_cstr));
    env->ReleaseStringUTFChars(prompt, prompt_cstr);
    return env->NewStringUTF(stub.c_str());
#endif
}

// ---------------------------------------------------------------------------
// JNI: nativeFreeModel
// ---------------------------------------------------------------------------

extern "C"
JNIEXPORT void JNICALL
Java_com_example_adaptillm_core_LlamaCppBridge_nativeFreeModel(
        JNIEnv *env,
        jobject /* this */) {

    LOGI("nativeFreeModel: releasing resources");

#ifdef LLAMA_AVAILABLE
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    if (g_backend_inited) {
        llama_backend_free();
        g_backend_inited = false;
    }
#endif

    g_model_loaded = false;
    g_ctx_size = 0;
    g_ctx_threads = 0;
    LOGI("nativeFreeModel: done");
}

// ---------------------------------------------------------------------------
// JNI: nativeGenerateStreaming
//
// THE MAIN INFERENCE PATH. Optimized:
//   1. Context REUSED (not rebuilt) if params match
//   2. Only KV-cache cleared per query (fast: ~0ms)
//   3. Stop flag checked every token for user abort
//   4. Sentence-boundary safe stopping at 80% token limit
// ---------------------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_adaptillm_core_LlamaCppBridge_nativeGenerateStreaming(
        JNIEnv *env,
        jobject /* this */,
        jstring prompt,
        jint maxTokens,
        jint threads,
        jint contextSize,
        jobject callback) {

    if (!g_model_loaded) {
        return env->NewStringUTF("ERROR: Model not loaded");
    }

    // Reset stop flag
    g_stop_requested.store(false);

    const char *prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
    if (prompt_cstr == nullptr) {
        return env->NewStringUTF("ERROR: null prompt");
    }

    LOGI("nativeGenerateStreaming: prompt_len=%zu, max_tokens=%d, threads=%d, ctx=%d",
         strlen(prompt_cstr), (int)maxTokens, (int)threads, (int)contextSize);

    // Get callback method ID
    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
    if (onTokenMethod == nullptr) {
        LOGE("Could not find onToken method");
        env->ReleaseStringUTFChars(prompt, prompt_cstr);
        return env->NewStringUTF("ERROR: callback method not found");
    }

#ifdef LLAMA_AVAILABLE

    // --- KEY OPTIMIZATION: Reuse context, only clear KV-cache ---
    if (!ensure_context(contextSize, threads)) {
        env->ReleaseStringUTFChars(prompt, prompt_cstr);
        return env->NewStringUTF("ERROR: Failed to create inference context");
    }

    // Clear KV-cache only (NOT the full context)
    // This takes ~0ms vs ~2-5 seconds for context recreation
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Tokenize
    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);

    int32_t n_prompt_tokens = llama_tokenize(
        vocab, prompt_cstr, (int32_t)strlen(prompt_cstr),
        nullptr, 0, true, false
    );
    if (n_prompt_tokens < 0) n_prompt_tokens = -n_prompt_tokens;

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    int32_t actual = llama_tokenize(
        vocab, prompt_cstr, (int32_t)strlen(prompt_cstr),
        prompt_tokens.data(), (int32_t)prompt_tokens.size(),
        true, false
    );

    env->ReleaseStringUTFChars(prompt, prompt_cstr);

    if (actual < 0) {
        return env->NewStringUTF("ERROR: Tokenization failed");
    }
    prompt_tokens.resize(actual);

    // Truncate if needed
    if (actual >= contextSize) {
        LOGW("Prompt (%d) >= context (%d), truncating", actual, contextSize);
        prompt_tokens.resize(contextSize - 1);
        actual = (int32_t)prompt_tokens.size();
    }

    LOGI("Streaming: %d prompt tokens, generating up to %d", actual, (int)maxTokens);

    // Prefill
    struct llama_batch batch = llama_batch_get_one(prompt_tokens.data(), actual);
    int32_t decode_result = llama_decode(g_ctx, batch);
    if (decode_result != 0) {
        return env->NewStringUTF("ERROR: Initial decode failed");
    }

    // Sampler — tight settings for focused generation
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.3f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(15));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.85f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.2f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    // Streaming generation loop with STOP support + stop-sequence detection
    std::string result;
    result.reserve(maxTokens * 8);
    int softLimit = (int)(maxTokens * 0.80);
    bool stop_seq_hit = false;

    for (int i = 0; i < maxTokens; i++) {
        // Check user stop flag
        if (g_stop_requested.load()) {
            LOGI("Generation stopped by user at step %d", i);
            break;
        }

        llama_token new_token = llama_sampler_sample(smpl, g_ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            LOGD("EOG at step %d", i);
            break;
        }

        llama_sampler_accept(smpl, new_token);

        char piece[256];
        int32_t n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, false);

        if (n_chars > 0) {
            result.append(piece, n_chars);

            // Stop-sequence detection BEFORE streaming to UI
            if (check_stop_sequences(result)) {
                LOGD("Stop-sequence break at step %d", i);
                stop_seq_hit = true;
                break;
            }

            // Stream clean token to Kotlin
            std::string token_str(piece, n_chars);
            jstring jtoken = env->NewStringUTF(token_str.c_str());
            env->CallVoidMethod(callback, onTokenMethod, jtoken);
            env->DeleteLocalRef(jtoken);
        }

        // Sentence-boundary safe stopping
        if (i >= softLimit && !result.empty()) {
            char lastChar = result.back();
            if (lastChar == '.' || lastChar == '!' || lastChar == '?' || lastChar == '\n') {
                LOGD("Safe stop at step %d/%d", i, maxTokens);
                break;
            }
        }

        struct llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("Decode failed at step %d", i);
            break;
        }
    }

    llama_sampler_free(smpl);
    LOGI("Streaming complete: %zu chars, stopped=%d", result.size(), g_stop_requested.load());
    return env->NewStringUTF(result.c_str());

#else
    env->ReleaseStringUTFChars(prompt, prompt_cstr);
    return env->NewStringUTF("[STUB] Streaming not available without llama.cpp");
#endif
}

// ---------------------------------------------------------------------------
// JNI: nativeStopGeneration
//
// Sets the stop flag. The generation loop checks this every token.
// ---------------------------------------------------------------------------

extern "C"
JNIEXPORT void JNICALL
Java_com_example_adaptillm_core_LlamaCppBridge_nativeStopGeneration(
        JNIEnv *env,
        jobject /* this */) {
    g_stop_requested.store(true);
    LOGI("Stop requested");
}

// ---------------------------------------------------------------------------
// JNI: nativeCacheSystemPrompt
// ---------------------------------------------------------------------------

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_adaptillm_core_LlamaCppBridge_nativeCacheSystemPrompt(
        JNIEnv *env,
        jobject /* this */,
        jstring sysPrompt,
        jint threads,
        jint contextSize) {

    if (!g_model_loaded) return JNI_FALSE;

    const char *sys_cstr = env->GetStringUTFChars(sysPrompt, nullptr);
    if (sys_cstr == nullptr) return JNI_FALSE;

#ifdef LLAMA_AVAILABLE
    if (!ensure_context(contextSize, threads)) {
        env->ReleaseStringUTFChars(sysPrompt, sys_cstr);
        return JNI_FALSE;
    }

    llama_memory_clear(llama_get_memory(g_ctx), true);

    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);
    int32_t n_tokens = llama_tokenize(vocab, sys_cstr, (int32_t)strlen(sys_cstr), nullptr, 0, true, false);
    if (n_tokens < 0) n_tokens = -n_tokens;

    std::vector<llama_token> tokens(n_tokens);
    int32_t actual = llama_tokenize(vocab, sys_cstr, (int32_t)strlen(sys_cstr),
        tokens.data(), (int32_t)tokens.size(), true, false);
    env->ReleaseStringUTFChars(sysPrompt, sys_cstr);

    if (actual < 0) return JNI_FALSE;
    tokens.resize(actual);

    struct llama_batch batch = llama_batch_get_one(tokens.data(), actual);
    if (llama_decode(g_ctx, batch) != 0) return JNI_FALSE;

    LOGI("System prompt cached: %d tokens", actual);
    return JNI_TRUE;
#else
    env->ReleaseStringUTFChars(sysPrompt, sys_cstr);
    return JNI_FALSE;
#endif
}

// ---------------------------------------------------------------------------
// JNI: nativeGenerateWithCache
// ---------------------------------------------------------------------------

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_adaptillm_core_LlamaCppBridge_nativeGenerateWithCache(
        JNIEnv *env,
        jobject thiz,
        jstring userPrompt,
        jint maxTokens,
        jint threads,
        jint contextSize,
        jobject callback) {

    // Falls back to full streaming if context params don't match
    if (!g_model_loaded || g_ctx == nullptr ||
        g_ctx_size != contextSize || g_ctx_threads != threads) {
        return Java_com_example_adaptillm_core_LlamaCppBridge_nativeGenerateStreaming(
            env, thiz, userPrompt, maxTokens, threads, contextSize, callback
        );
    }

    g_stop_requested.store(false);

    const char *user_cstr = env->GetStringUTFChars(userPrompt, nullptr);
    if (user_cstr == nullptr) return env->NewStringUTF("ERROR: null prompt");

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");

#ifdef LLAMA_AVAILABLE
    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);

    int32_t n_user = llama_tokenize(vocab, user_cstr, (int32_t)strlen(user_cstr), nullptr, 0, false, false);
    if (n_user < 0) n_user = -n_user;

    std::vector<llama_token> user_tokens(n_user);
    int32_t actual = llama_tokenize(vocab, user_cstr, (int32_t)strlen(user_cstr),
        user_tokens.data(), (int32_t)user_tokens.size(), false, false);
    env->ReleaseStringUTFChars(userPrompt, user_cstr);

    if (actual < 0) return env->NewStringUTF("ERROR: Tokenization failed");
    user_tokens.resize(actual);

    LOGI("CacheGen: %d user tokens (sys prompt already in KV)", actual);

    struct llama_batch batch = llama_batch_get_one(user_tokens.data(), actual);
    if (llama_decode(g_ctx, batch) != 0) {
        return env->NewStringUTF("ERROR: User decode failed");
    }

    // Sampler + generation (same as streaming)
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.3f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.8f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.3f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    std::string result;
    result.reserve(maxTokens * 8);
    int softLimit = (int)(maxTokens * 0.80);

    for (int i = 0; i < maxTokens; i++) {
        if (g_stop_requested.load()) break;

        llama_token new_token = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        llama_sampler_accept(smpl, new_token);

        char piece[256];
        int32_t n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, false);
        if (n_chars > 0) {
            result.append(piece, n_chars);
            if (onTokenMethod != nullptr) {
                std::string token_str(piece, n_chars);
                jstring jtoken = env->NewStringUTF(token_str.c_str());
                env->CallVoidMethod(callback, onTokenMethod, jtoken);
                env->DeleteLocalRef(jtoken);
            }
        }

        if (i >= softLimit && !result.empty()) {
            char lastChar = result.back();
            if (lastChar == '.' || lastChar == '!' || lastChar == '?' || lastChar == '\n') break;
        }

        struct llama_batch next = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, next) != 0) break;
    }

    llama_sampler_free(smpl);
    LOGI("CacheGen complete: %zu chars", result.size());
    return env->NewStringUTF(result.c_str());

#else
    env->ReleaseStringUTFChars(userPrompt, user_cstr);
    return env->NewStringUTF("[STUB] Cache gen not available");
#endif
}

// ---------------------------------------------------------------------------
// JNI: nativeInvalidateCache
// ---------------------------------------------------------------------------

extern "C"
JNIEXPORT void JNICALL
Java_com_example_adaptillm_core_LlamaCppBridge_nativeInvalidateCache(
        JNIEnv *env,
        jobject /* this */) {
#ifdef LLAMA_AVAILABLE
    if (g_ctx != nullptr) {
        llama_memory_clear(llama_get_memory(g_ctx), true);
        LOGI("KV cache cleared");
    }
#endif
}
