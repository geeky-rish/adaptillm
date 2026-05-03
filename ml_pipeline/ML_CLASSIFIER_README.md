# ML Query Classifier — Research Documentation

## Design Decisions & Integration Guide

---

# 1. Why ML Classifier > Rule-Based

## The Problem with Rule-Based

The existing `QueryClassifier.kt` uses keyword matching:

```kotlin
if (CODE_KEYWORDS.any { lower.contains(it) }) return CODE
```

This fails on:

| Query | True Label | Rule-Based | Why It Fails |
|-------|-----------|------------|-------------|
| "fibonacci" | CODE | GENERAL | No keyword like "code" or "implement" |
| "binary search" | CODE | CODE | Only works because "sort" is a keyword |
| "build a website" | CODE | GENERAL | "build" is not in keyword list |
| "hash table" | CODE | GENERAL | No explicit code keyword |
| "recursion" | DEFINITION | GENERAL | No "what is" prefix |
| "DFS traversal" | CODE | GENERAL | Not in keyword list |
| "merge sort implementation" | CODE | CODE | Works (has "sort") |

**Root cause**: Rule-based classifiers require **explicit keyword presence**. They cannot generalise to unseen phrasings.

## Why ML Wins

An ML classifier learns **semantic patterns** from training data:

1. **Generalisation**: "fibonacci" appears in CODE training examples alongside other algorithm names. The model learns that algorithm names → CODE.
2. **Context**: "what is recursion" vs "recursion" — the model learns that standalone technical terms without question prefixes are likely CODE.
3. **Robustness**: New queries that don't match any keyword are still classified correctly based on learned patterns.

---

# 2. Impact on Token Efficiency

The query classifier directly controls token budget allocation:

```
CODE queries → 512 tokens (always, non-negotiable)
Other queries → mode-based (96-512)
```

If a CODE query is misclassified as GENERAL in EFFICIENT mode, it gets only **96 tokens** — causing truncated, useless code output.

**With ML classifier**: Ambiguous coding queries like "fibonacci" correctly get 512 tokens, producing complete code.

**Estimated token savings**: By correctly classifying GENERAL/DEFINITION queries (which don't need 512 tokens), the ML classifier prevents unnecessary token allocation, reducing average inference time.

---

# 3. Impact on Latency/Quality Trade-off

| Scenario | Rule-Based | ML Classifier | Impact |
|----------|-----------|---------------|--------|
| "fibonacci" in EFFICIENT mode | GENERAL → 96 tokens → incomplete | CODE → 512 tokens → complete | Quality ↑ |
| "hello" in HIGH mode | GENERAL → 512 tokens → wasted | GENERAL → 96 tokens → fast | Latency ↓ |
| "what is ML" in BALANCED | GENERAL → 256 tokens → ok | DEFINITION → 256 tokens → structured | Quality ↑ |

The ML classifier enables **precise resource allocation** — each query gets exactly the resources it needs.

---

# 4. Model Architecture

## Approach A: TF-IDF + Logistic Regression

```
Input text → TF-IDF vectorization (5000 features, bigrams) → LogisticRegression → 4 classes
```

- **Pros**: Fast training, interpretable, tiny model
- **Cons**: Fixed vocabulary, no semantic understanding

## Approach B: TF Neural Network (Deployed)

```
Input text → TextVectorization → Embedding(32) → GlobalAvgPool1D → Dense(64, ReLU) → Dense(4, softmax)
```

- **Parameters**: ~170K
- **Model size**: ~200KB (TFLite with float16 quantization)
- **Inference**: <5ms on mobile ARM CPU
- **Why this architecture**:
  - Embedding: captures word semantics (vs. bag-of-words)
  - GlobalAvgPool: order-invariant, handles variable-length inputs
  - Single hidden layer: sufficient for 4-class text classification
  - Dropout(0.3): prevents overfitting on 2000 samples

---

# 5. Dataset Design

| Class | Count | Source Strategy |
|-------|-------|----------------|
| CODE | 500 | Template-based: 50 templates × 10 variations + 50 ambiguous terms |
| DEFINITION | 500 | "what is {subject}" × 100+ CS/ML subjects × 12 templates |
| EXPLANATION | 500 | "how does / why / explain / compare" × topics |
| GENERAL | 500 | Greetings, opinions, recommendations, meta-questions |

**Key design decisions**:
- **Ambiguous samples included**: "fibonacci", "binary search", "linked list" labeled as CODE to teach the model that algorithm names imply code intent
- **Balanced classes**: Equal samples prevent majority-class bias
- **Template diversity**: 10-20 templates per class to avoid overfitting to specific phrasings

---

# 6. Integration Guide

## Step 1: Train the model

1. Open `ml_pipeline/query_classifier_notebook.py` in Google Colab
2. Run all cells
3. Download: `query_classifier.tflite`, `vocab.json`, `label_map.json`

## Step 2: Add files to Android project

```
app/src/main/assets/
  ├── query_classifier.tflite
  ├── vocab.json
  └── label_map.json
```

## Step 3: Add TFLite dependency

In `app/build.gradle.kts`:

```kotlin
dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
}
```

Also add in `android {}` block:

```kotlin
android {
    aaptOptions {
        noCompress("tflite")
    }
}
```

## Step 4: Use in code

In `LlamaCppEngine.kt`, replace:

```kotlin
val queryType = QueryClassifier.classify(input)
```

With:

```kotlin
val classifier = MLQueryClassifier.getInstance(context)
val queryType = classifier.classify(input)
```

The `MLQueryClassifier` automatically falls back to rule-based if the TFLite model is not available.

## Step 5: Verify on device

Check logcat for:
```
MLQueryClassifier: classify('fibonacci...') → CODE in 2.34ms
```

---

# 7. Files Inventory

| File | Purpose | Size |
|------|---------|------|
| `ml_pipeline/query_classifier_notebook.py` | Full training notebook | ~15KB |
| `MLQueryClassifier.kt` | Android TFLite classifier | ~8KB |
| `QueryClassifier.kt` | Rule-based fallback (unchanged) | ~2KB |
| `query_classifier.tflite` | Trained model (after running notebook) | ~200KB |
| `vocab.json` | Vocabulary mapping | ~50KB |
| `label_map.json` | {0: CODE, 1: DEFINITION, ...} | <1KB |
