# AdaptiLLM 🧠⚡

**Adaptive On-Device LLM Inference Engine for Android**

An Android application that runs a quantised large language model (1.5B parameters) entirely on-device using `llama.cpp`, with an adaptive inference pipeline that dynamically adjusts generation parameters based on real-time device telemetry.

> No cloud. No API keys. No internet required. Runs on CPU-only Android devices.

---

## What This Does

Most on-device LLM systems use **fixed configurations** — same thread count, same context window, same token budget for every query. This wastes resources: a simple "what is recursion?" doesn't need the same compute as "write a merge sort in C++".

AdaptiLLM solves this with a **6-stage adaptive inference pipeline**:

```
User Input
    ↓
┌─────────────┐
│ QueryRouter  │ ── Greetings? → Instant deterministic response (0ms)
└──────┬──────┘
       ↓
┌──────────────────┐
│ ML Classifier    │ ── TFLite model classifies: CODE / DEFINITION / EXPLANATION / GENERAL
└──────┬───────────┘
       ↓
┌──────────────────────┐
│ Adaptive Policy      │ ── Softmax-weighted cost function over latency/energy/quality
│ Engine               │    selects: HIGH / BALANCED / EFFICIENT mode
└──────┬───────────────┘
       ↓
┌──────────────────┐
│ LlamaCppEngine   │ ── Native C++ inference via JNI with KV-cache reuse
└──────┬───────────┘
       ↓
┌──────────────────┐
│ OutputCleaner    │ ── 8-pass sanitisation: stop-sequence, hallucination, length, code validation
└──────┬───────────┘
       ↓
┌──────────────────┐
│ MetricsLogger    │ ── Per-inference CSV: latency, TTFT, energy proxy (µA·ms), tokens/s
└──────────────────┘
```

---

## Technical Highlights

### Adaptive Policy Engine (`AdaptivePolicyEngine.kt`)
- **Multi-objective optimisation**: Minimises composite cost `C(m) = w_L·L̂(m) + w_E·Ê(m) + w_Q·Q̂(m)`
- **Softmax-normalised weights**: Derived from battery pressure (sigmoid at 30%), memory pressure (sigmoid at 65%), and device health
- **EMA feedback loop**: Running statistics updated after each inference (α=0.3)
- **No hard thresholds**: Smooth, differentiable mode transitions

### Query Router (`QueryRouter.kt`)
- **26 deterministic greeting responses** — zero latency, no model call
- **Type-specific system prompts** (≤15 words each) to prevent instruction bleed
- **Proportional token budgets**: CODE=512, DEFINITION=96, EXPLANATION=160, GENERAL=128
- **Complexity scaling**: TRIVIAL→24, SHORT→48, MEDIUM→128, COMPLEX→256 tokens

### Native C++ Layer (`llama_interface.cpp`)
- **11 stop-sequence patterns** checked every token — prevents template leakage at source
- **Partial stop detection** — holds back streaming tokens that could start a stop sequence
- **Soft stopping** — monitors for sentence boundaries after 80% token budget
- **Atomic interruption** — user can cancel generation mid-stream

### Output Cleaner (`OutputCleaner.kt`)
8-pass sanitisation pipeline:
1. Template marker truncation
2. Fragment removal
3. Instruction echo detection
4. Hallucination marker detection
5. Length enforcement (600 chars text / 1500 chars code)
6. Sentence boundary trimming
7. Code validation (strips markdown fences, verifies code indicators)
8. Nonsense detection with type-specific fallbacks

### Energy Measurement (`DeviceMonitor.kt`)
- Battery % has integer granularity (always Δ=0 for fast inferences)
- Uses `BATTERY_PROPERTY_CURRENT_NOW` for µA-resolution current sampling
- Energy proxy: `avg(µA_before, µA_after) × latency_ms` — proportional to joules

---

## Architecture

```
app/src/main/
├── java/com/example/adaptillm/
│   ├── core/                          # Engine layer
│   │   ├── AdaptivePolicyEngine.kt    # Multi-objective mode selection
│   │   ├── QueryRouter.kt            # Input routing + greeting bypass
│   │   ├── QueryClassifier.kt        # Unified classification entry point
│   │   ├── QueryClassifierML.kt      # TFLite-based ML classifier
│   │   ├── MLQueryClassifier.kt      # ML model wrapper
│   │   ├── LlamaCppEngine.kt         # Inference orchestration
│   │   ├── LlamaCppBridge.kt         # JNI bridge to native code
│   │   ├── OutputCleaner.kt          # 8-pass output sanitisation
│   │   ├── ConversationManager.kt    # Multi-turn memory with token budget
│   │   ├── DeviceMonitor.kt          # Battery + memory telemetry
│   │   ├── MetricsLogger.kt          # Per-inference CSV logging
│   │   ├── InferenceController.kt    # Mode definitions
│   │   ├── ModeTracker.kt            # Mode history tracking
│   │   └── BenchmarkRunner.kt        # Automated benchmark suite
│   └── ui/
│       └── MainActivity.kt           # UI + pipeline orchestration
├── cpp/
│   ├── llama_interface.cpp            # Native inference + stop-sequence detection
│   └── CMakeLists.txt                 # NDK build config
└── res/                               # Android resources
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language (App) | Kotlin |
| Language (Native) | C++ (NDK) |
| LLM Runtime | [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI |
| ML Classifier | TensorFlow Lite |
| Model Format | GGUF (Q4_K_M quantisation) |
| Build System | Gradle + CMake |
| Target | Android 8.0+ (API 26+) |

---

## How to Run

### Prerequisites
- Android Studio with NDK installed
- Android device with USB debugging enabled
- A GGUF model file (e.g., [Qwen2.5-1.5B-Instruct Q4_K_M](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF))

### Build & Deploy
```bash
# 1. Clone
git clone https://github.com/geeky-rish/adaptillm.git
cd adaptillm

# 2. Build
./gradlew assembleDebug

# 3. Push model to device
adb shell mkdir -p /sdcard/Android/data/com.example.adaptillm/files/llama-model
adb push Qwen2.5-1.5B-Instruct-Q4_K_M.gguf /sdcard/Android/data/com.example.adaptillm/files/llama-model/

# 4. Install
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## Key Design Decisions

1. **Why adaptive?** Static configurations waste resources on simple queries and under-serve complex ones. The softmax-weighted policy smoothly rebalances priorities as battery drains and memory fills.

2. **Why greeting bypass?** Small models (1-3B) produce verbose, hallucinatory responses for trivial inputs like "hi". Deterministic bypass eliminates this failure mode entirely.

3. **Why 4-layer defence?** No single sanitisation layer catches everything. C++ catches template tokens before streaming; Kotlin catches instruction bleed and hallucinations; UI replaces streamed text with the cleaned version.

4. **Why µA energy proxy?** Battery percentage (0-100) has zero resolution for 1-10 second inferences. Microampere current sampling provides real energy signal for optimisation.

5. **Why ≤15 word system prompts?** Small models echo long instructions. Minimal prompts prevent "instruction bleed" where the model paraphrases its own system prompt.

---

## What I Built vs. What's External

| Component | Built by Me | External |
|-----------|------------|----------|
| Adaptive policy engine (softmax cost function) | ✅ | |
| Query router + greeting bypass | ✅ | |
| Output cleaner (8-pass pipeline) | ✅ | |
| C++ stop-sequence detection + soft stopping | ✅ | |
| JNI bridge layer | ✅ | |
| ML query classifier training + integration | ✅ | |
| Energy proxy measurement | ✅ | |
| Conversation memory with token budget | ✅ | |
| llama.cpp inference runtime | | ✅ (open source) |
| TensorFlow Lite runtime | | ✅ (Google) |
| Android SDK / NDK | | ✅ (Google) |

---

## Metrics Collected Per Inference

| Metric | Unit | Purpose |
|--------|------|---------|
| `latency_ms` | ms | End-to-end inference time |
| `ttft_ms` | ms | Time to first token |
| `energy_proxy_uams` | µA·ms | Energy consumption proxy |
| `tokens_per_sec` | tok/s | Generation throughput |
| `mode` | enum | HIGH / BALANCED / EFFICIENT |
| `query_type` | enum | CODE / DEFINITION / EXPLANATION / GENERAL |
| `complexity` | enum | TRIVIAL / SHORT / MEDIUM / COMPLEX |
| `was_greeting` | bool | Deterministic bypass triggered |
| `context_size` | int | Active context window |
| `max_tokens` | int | Token budget for this query |

---

## License

This project is for academic and demonstration purposes.

---

*Built with Kotlin, C++, and a lot of stop-sequence debugging.* 🛠️
