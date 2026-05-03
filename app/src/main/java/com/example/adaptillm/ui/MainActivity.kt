package com.example.adaptillm.ui

import com.example.adaptillm.R
import com.example.adaptillm.core.AdaptivePolicyEngine
import com.example.adaptillm.core.CodeExtractor
import com.example.adaptillm.core.DeviceMonitor
import com.example.adaptillm.core.LlamaCppEngine
import com.example.adaptillm.core.LLMInterface
import com.example.adaptillm.core.MetricsLogger
import com.example.adaptillm.core.ModeTracker
import com.example.adaptillm.core.QueryClassifier
import com.example.adaptillm.core.QueryClassifierML
import com.example.adaptillm.core.QueryRouter

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.app.Activity
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast

/**
 * MainActivity -- Stabilized inference pipeline with:
 *   - Greeting bypass (deterministic, no LLM)
 *   - µA-based energy measurement (real delta for Pareto)
 *   - Accurate config logging (actual values, not hardcoded)
 *   - Streaming with TTFT tracking
 */
class MainActivity : Activity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    // --- UI ---
    private lateinit var etInput: EditText
    private lateinit var btnGenerate: Button
    private lateinit var btnStop: Button
    private lateinit var btnNewChat: Button
    private lateinit var btnCopy: Button
    private lateinit var btnCopyCode: Button
    private lateinit var copyButtonsRow: LinearLayout
    private lateinit var tvOutput: TextView
    private lateinit var tvStatus: TextView

    // --- Core ---
    private lateinit var deviceMonitor: DeviceMonitor
    private lateinit var policyEngine: AdaptivePolicyEngine
    private lateinit var llmEngine: LlamaCppEngine
    private lateinit var metricsLogger: MetricsLogger
    private lateinit var modeTracker: ModeTracker

    // --- State ---
    private var lastLatencyMs: Long = -1
    private var lastResponseLength: Int = -1
    private var streamingTokenCount = 0
    private var firstTokenTimeMs: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initViews()
        initCoreModules()
        setupListeners()
        displayInitialStatus()
    }

    override fun onDestroy() {
        super.onDestroy()
        llmEngine.shutdown()
    }

    private fun initViews() {
        etInput = findViewById(R.id.etInput)
        btnGenerate = findViewById(R.id.btnGenerate)
        btnStop = findViewById(R.id.btnStop)
        btnNewChat = findViewById(R.id.btnNewChat)
        btnCopy = findViewById(R.id.btnCopy)
        btnCopyCode = findViewById(R.id.btnCopyCode)
        copyButtonsRow = findViewById(R.id.copyButtonsRow)
        tvOutput = findViewById(R.id.tvOutput)
        tvStatus = findViewById(R.id.tvStatus)
    }

    private fun initCoreModules() {
        deviceMonitor = DeviceMonitor()
        policyEngine = AdaptivePolicyEngine()
        llmEngine = LlamaCppEngine()
        metricsLogger = MetricsLogger()
        modeTracker = ModeTracker()

        metricsLogger.init(this)

        val mlReady = QueryClassifierML.init(this)
        Log.d(TAG, "Init | ML:${if (mlReady) "ready" else "rules"} | CSV:${metricsLogger.csvPath()}")
    }

    private fun setupListeners() {
        btnGenerate.setOnClickListener { handleStreamingInference() }
        btnStop.setOnClickListener { handleStop() }
        btnNewChat.setOnClickListener { handleNewChat() }
        btnCopy.setOnClickListener { copyToClipboard(tvOutput.text.toString(), "Output") }
        btnCopyCode.setOnClickListener { copyCodeToClipboard() }
    }

    private fun displayInitialStatus() {
        val battery = deviceMonitor.getBatteryLevel(this)
        val memory = deviceMonitor.getMemoryUsage(this)
        tvStatus.text = String.format("Ready | Battery: %d%% | Memory: %.0f%% | Turn: 0", battery, memory)
    }

    // ---------------------------------------------------------------
    // Stop / New Chat
    // ---------------------------------------------------------------

    private fun handleStop() {
        llmEngine.stopGeneration()
        tvStatus.text = "Stopped"
        Toast.makeText(this, "Generation stopped", Toast.LENGTH_SHORT).show()
    }

    private fun handleNewChat() {
        llmEngine.resetConversation()
        tvOutput.text = "Output will appear here..."
        copyButtonsRow.visibility = View.GONE
        val battery = deviceMonitor.getBatteryLevel(this)
        val memory = deviceMonitor.getMemoryUsage(this)
        tvStatus.text = String.format("New Chat | Battery: %d%% | Memory: %.0f%% | Turn: 0", battery, memory)
        Toast.makeText(this, "Conversation cleared", Toast.LENGTH_SHORT).show()
    }

    // ---------------------------------------------------------------
    // Clipboard
    // ---------------------------------------------------------------

    private fun copyToClipboard(text: String, label: String) {
        if (text.isEmpty() || text == "Output will appear here..." || text.startsWith("\u23F3")) {
            Toast.makeText(this, "Nothing to copy", Toast.LENGTH_SHORT).show()
            return
        }
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("AdaptiLLM $label", text))
        Toast.makeText(this, "$label copied", Toast.LENGTH_SHORT).show()
    }

    private fun copyCodeToClipboard() {
        val output = tvOutput.text.toString()
        val code = CodeExtractor.extractCode(output)
        if (code.isEmpty()) {
            Toast.makeText(this, "No code block found", Toast.LENGTH_SHORT).show()
        } else {
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            clipboard.setPrimaryClip(ClipData.newPlainText("AdaptiLLM Code", code))
            Toast.makeText(this, "Code copied", Toast.LENGTH_SHORT).show()
        }
    }

    // ---------------------------------------------------------------
    // Streaming Inference Pipeline
    // ---------------------------------------------------------------

    private fun handleStreamingInference() {
        val userInput = etInput.text.toString().trim()
        if (userInput.isEmpty()) {
            tvOutput.text = "Error: Input is empty"
            return
        }

        setLoadingState(true)
        tvOutput.text = "\u23F3 Generating..."
        copyButtonsRow.visibility = View.GONE
        streamingTokenCount = 0
        firstTokenTimeMs = 0

        val queryType = QueryClassifier.classify(userInput)
        val batteryBefore = deviceMonitor.getBatteryLevel(this)
        val currentUaBefore = deviceMonitor.getBatteryMicroAmps(this)
        val memoryUsage = deviceMonitor.getMemoryUsage(this)
        val turnsBefore = llmEngine.conversationManager.turnCount()

        val snapshot = AdaptivePolicyEngine.SystemSnapshot(
            batteryPercent = batteryBefore,
            memoryUsagePercent = memoryUsage,
            lastLatencyMs = lastLatencyMs,
            lastResponseLength = lastResponseLength
        )
        val selectedMode = policyEngine.selectMode(snapshot)

        tvStatus.text = String.format(
            "\u26A1 %s | %s | Batt:%d%% | Turn:%d | Streaming...",
            selectedMode, queryType.name, batteryBefore, turnsBefore
        )

        val startTimeMs = System.currentTimeMillis()

        llmEngine.generateStreaming(
            input = userInput,
            mode = selectedMode,
            onToken = { token ->
                if (streamingTokenCount == 0) {
                    tvOutput.text = ""
                    firstTokenTimeMs = System.currentTimeMillis()
                    val ttft = firstTokenTimeMs - startTimeMs
                    tvStatus.text = String.format(
                        "\u26A1 %s | %s | TTFT: %dms | Turn:%d | Streaming...",
                        selectedMode, queryType.name, ttft, turnsBefore
                    )
                }
                streamingTokenCount++
                tvOutput.append(token)
            },
            onComplete = { fullResponse ->
                val endTimeMs = System.currentTimeMillis()
                val latencyMs = endTimeMs - startTimeMs
                val ttft = if (firstTokenTimeMs > 0) firstTokenTimeMs - startTimeMs else latencyMs

                // Battery measurements (both % and µA)
                val batteryAfter = deviceMonitor.getBatteryLevel(this)
                val currentUaAfter = deviceMonitor.getBatteryMicroAmps(this)
                val batteryDelta = (batteryBefore - batteryAfter).coerceAtLeast(0)
                val energyProxy = deviceMonitor.computeEnergyProxy(currentUaBefore, currentUaAfter, latencyMs)

                lastLatencyMs = latencyMs
                lastResponseLength = fullResponse.length

                policyEngine.recordOutcome(selectedMode, latencyMs, fullResponse.length)
                modeTracker.record(selectedMode, batteryBefore, memoryUsage, queryType.name, latencyMs)

                val tokensPerSec = if (latencyMs > 0) (streamingTokenCount * 1000.0 / latencyMs) else 0.0
                val turnsNow = llmEngine.conversationManager.turnCount()

                // Get actual config from engine (not hardcoded!)
                val actualConfig = llmEngine.lastConfig
                val wasGreeting = llmEngine.lastWasGreeting
                val complexity = llmEngine.lastComplexity

                // --- Log to CSV with real values ---
                metricsLogger.log(MetricsLogger.InferenceRecord(
                    queryText = userInput,
                    queryType = queryType.name,
                    mode = selectedMode,
                    complexity = complexity.name,
                    latencyMs = latencyMs,
                    ttftMs = ttft,
                    batteryStart = batteryBefore,
                    batteryEnd = batteryAfter,
                    batteryDelta = batteryDelta,
                    currentUaBefore = currentUaBefore,
                    currentUaAfter = currentUaAfter,
                    energyProxyUaMs = energyProxy,
                    tokensGenerated = streamingTokenCount,
                    tokensPerSec = tokensPerSec,
                    responseLength = fullResponse.length,
                    contextSize = actualConfig.contextSize,
                    maxTokens = actualConfig.maxTokens,
                    numThreads = actualConfig.numThreads,
                    wasTruncated = false,
                    wasGreeting = wasGreeting,
                    memoryPercent = memoryUsage,
                    timestamp = startTimeMs
                ))

                // Display sanitized response (replace streamed content)
                tvOutput.text = fullResponse

                tvStatus.text = String.format(
                    "%s | %s | %dms | TTFT:%dms | %.1f tok/s | Batt:%d%% | Turn:%d",
                    selectedMode, queryType.name, latencyMs, ttft, tokensPerSec,
                    batteryAfter, turnsNow
                )

                if (fullResponse.isNotEmpty() && !fullResponse.startsWith("ERROR") &&
                    !fullResponse.startsWith("Model not") && !fullResponse.startsWith("Native")) {
                    copyButtonsRow.visibility = View.VISIBLE
                }

                setLoadingState(false)

                Log.d(TAG, "Pipeline | mode=$selectedMode type=$queryType " +
                        "total=${latencyMs}ms ttft=${ttft}ms tokens=$streamingTokenCount " +
                        "tok/s=${String.format("%.1f", tokensPerSec)} turns=$turnsNow " +
                        "greeting=$wasGreeting energy=${energyProxy}uA*ms")
            }
        )
    }

    // ---------------------------------------------------------------
    // UI helpers
    // ---------------------------------------------------------------

    private fun setLoadingState(isLoading: Boolean) {
        btnGenerate.isEnabled = !isLoading
        btnGenerate.visibility = if (isLoading) View.GONE else View.VISIBLE
        btnStop.visibility = if (isLoading) View.VISIBLE else View.GONE
        etInput.isEnabled = !isLoading
        btnNewChat.isEnabled = !isLoading
    }
}