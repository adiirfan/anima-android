package com.anima.npu

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID
import kotlin.random.Random

/** What the chat is doing now. phase: idle, waiting (for a picture), loading (Ling), thinking, writing, tool. */
data class TurnState(val convId: String?, val phase: String, val partialText: String, val partialThinking: String)

/**
 * One chat turn: save the user's message, wait for the AI chip, let Ling answer (streamed), and run any draw_picture
 * calls by adding jobs to the picture queue, then let Ling say one more line about it.
 */
class ChatController(ctx: Context) {
    private val app = ctx.applicationContext
    private val store = ChatStore(File(app.filesDir, "chats"))
    private val client = ChatClient()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val _state = MutableStateFlow(TurnState(null, "idle", "", ""))
    val state: StateFlow<TurnState> = _state
    /** Bumped whenever a conversation file changes, so screens reload it. */
    val changed = MutableStateFlow(0)
    @Volatile private var stopping = false

    fun store() = store
    fun busy() = _state.value.phase != "idle"

    /** Starts a turn; returns the conversation id (a new one when [convId] is null). */
    fun send(convId: String?, text: String, image: Bitmap?, thinking: Boolean): String {
        val id = convId ?: UUID.randomUUID().toString().take(12)
        val msgId = newId()
        val imageFile = image?.let { b ->
            val name = "$id-$msgId.png"
            File(store.dir(), name).outputStream().use { b.compress(Bitmap.CompressFormat.PNG, 100, it) }
            name
        }
        val prev = store.get(id)
        val title = prev?.title ?: text.lineSequence().first().take(40).ifBlank { "Picture" }
        save(Conversation(id, title, System.currentTimeMillis(), prev?.thinking ?: thinking,
            (prev?.msgs ?: emptyList()) + Msg.User(msgId, text, imageFile)))
        stopping = false
        _state.value = TurnState(id, "waiting", "", "")
        // started now, while the app is on screen: Android refuses it later from the background
        QueueService.kick(app)
        scope.launch { runCatching { turn(id) }.onFailure { e ->
            Log.e("ling", "turn failed", e)
            append(id, Msg.Assistant(newId(), "Something went wrong: ${e.message}", ""))
        }
            _state.value = TurnState(id, "idle", "", "")
        }
        return id
    }

    /** Stops the reply being written; what arrived so far is kept. */
    fun stop() { stopping = true }

    fun setThinking(convId: String, on: Boolean) {
        store.get(convId)?.let { save(it.copy(thinking = on)) }
    }

    private suspend fun turn(id: String) {
        // a picture already drawing finishes first (the gate puts the chat before the queue's next one)
        NpuGate.gate.withChat {
            if (!LocalLing.running()) {
                _state.value = TurnState(id, "loading", "", "")
                if (!LocalLing.start(app)) {
                    append(id, Msg.Assistant(newId(), "Ling could not start: ${LocalLing.lastLog}", ""))
                    return@withChat
                }
            }
            var badInARow = 0
            var queued = false
            for (round in 0 until 3) {
                val c = store.get(id) ?: return@withChat
                val calls = runRound(id, c) ?: return@withChat
                if (calls.isEmpty()) break
                if (round == 2) break  // two tool rounds per message at most
                val (ok, lastBad) = handleTools(id, calls)
                if (ok) { queued = true; badInARow = 0 } else badInARow++
                if (badInARow >= 2) {
                    append(id, Msg.Assistant(newId(), "I couldn't queue that picture: $lastBad", ""))
                    break
                }
                if (stopping) break
            }
            if (queued) QueueService.kick(app)
        }
    }

    private data class Call(var id: String = "", var name: String = "", val args: StringBuilder = StringBuilder())

    /** Streams one reply. Returns its tool calls (empty when it just answered), or null when stopped or failed. */
    private suspend fun runRound(id: String, c: Conversation): List<Call>? {
        val text = StringBuilder()
        val thought = StringBuilder()
        val calls = sortedMapOf<Int, Call>()
        _state.value = TurnState(id, if (c.thinking) "thinking" else "writing", "", "")
        try {
            client.stream(buildMessages(c), JSONArray().put(JSONObject(DRAW_TOOL_JSON)), c.thinking, { stopping }) { e ->
                when (e) {
                    is Event.Thinking -> { thought.append(e.delta); _state.value = TurnState(id, "thinking", text.toString(), thought.toString()) }
                    is Event.Text -> { text.append(e.delta); _state.value = TurnState(id, "writing", text.toString(), thought.toString()) }
                    is Event.Tool -> {
                        val k = calls.getOrPut(e.index) { Call() }
                        e.id?.let { k.id = it }; e.name?.let { k.name = it }; k.args.append(e.argsDelta)
                        _state.value = TurnState(id, "tool", text.toString(), thought.toString())
                    }
                    Event.Done -> {}
                }
            }
        } catch (e: Exception) {
            append(id, Msg.Assistant(newId(), text.toString().ifBlank { "Ling stopped answering: ${e.message}" }, thought.toString()))
            return null
        }
        if (text.isNotBlank() || thought.isNotBlank()) append(id, Msg.Assistant(newId(), text.toString().trim(), thought.toString().trim()))
        if (stopping) return null
        return calls.values.map { if (it.id.isEmpty()) it.also { k -> k.id = "call_" + newId() } else it }
    }

    /** Runs each draw_picture call. Returns (any queued, last error message). */
    private fun handleTools(id: String, calls: List<Call>): Pair<Boolean, String> {
        var ok = false
        var lastBad = ""
        val c = store.get(id) ?: return false to ""
        val lastImage = c.msgs.lastOrNull { it is Msg.User }?.let { (it as Msg.User).imageFile }
        for (k in calls) {
            append(id, Msg.ToolCall(newId(), k.id, k.name, k.args.toString()))
            val result = if (k.name != "draw_picture") JSONObject().put("error", "Unknown tool ${k.name}; use draw_picture.")
            else when (val r = checkDraw(k.args.toString(), lastImage != null, ModelStore.QWEN.installed(app) == "standard",
                    ModelStore.PICTURE.filter { it.installed(app) != null }.map { it.id }.toSet())) {
                is ToolCheck.Ok -> {
                    val jobs = enqueueFromChat(id, r.args, if (r.args.edit) lastImage else null)
                    jobs.forEach { j -> append(id, Msg.Picture(newId(), j.id)) }
                    ok = true
                    JSONObject().put("queued", jobs.size).put("position", Queue.store.waitingCount())
                }
                is ToolCheck.Bad -> { lastBad = r.message; JSONObject().put("error", r.message) }
            }
            append(id, Msg.ToolResult(newId(), k.id, result.toString()))
        }
        return ok to lastBad
    }

    private fun enqueueFromChat(convId: String, a: DrawArgs, image: String?): List<Job> {
        val drawing = app.getSharedPreferences("anima", Context.MODE_PRIVATE).getString("qwenDrawing", "qwen") ?: "qwen"
        val shape = when (a.size) { "portrait" -> 1; "landscape" -> 2; else -> 0 }
        return (1..a.count).map {
            val jid = UUID.randomUUID().toString().take(12)
            // the attached picture is copied beside the queue file, as the Create screen does
            val src = image?.let { name ->
                val dst = "$jid-src.png"
                File(store.dir(), name).copyTo(File(Queue.dir(app), dst), overwrite = true)
                dst
            }
            val j = Job(jid, System.currentTimeMillis(), a.model, drawing, a.prompt, Random.nextLong(0, 1_000_000_000),
                8, if (a.model == "anima") 1 else 0, shape, 1f, 0.5f, src, "chat:$convId")
            Queue.store.add(j)
            j
        }
    }

    private fun buildMessages(c: Conversation): JSONArray {
        val drawing = app.getSharedPreferences("anima", Context.MODE_PRIVATE).getString("qwenDrawing", "qwen")
        val a = JSONArray().put(JSONObject().put("role", "system")
            .put("content", systemPrompt(if (drawing == "noctq") "Noct Q" else "Qwen-Image")))
        for (m in c.msgs) when (m) {
            is Msg.User -> a.put(JSONObject().put("role", "user")
                .put("content", m.text + if (m.imageFile != null) "\n[the user attached a picture]" else ""))
            is Msg.Assistant -> if (m.text.isNotBlank()) a.put(JSONObject().put("role", "assistant").put("content", m.text))
            is Msg.ToolCall -> a.put(JSONObject().put("role", "assistant").put("content", "").put("tool_calls", JSONArray().put(
                JSONObject().put("id", m.callId).put("type", "function")
                    .put("function", JSONObject().put("name", m.name).put("arguments", m.argsJson)))))
            is Msg.ToolResult -> a.put(JSONObject().put("role", "tool").put("tool_call_id", m.callId).put("content", m.resultJson))
            is Msg.Picture -> {}  // the tool's answer already says it was queued
        }
        return a
    }

    private fun append(id: String, m: Msg) {
        val c = store.get(id) ?: return
        save(c.copy(msgs = c.msgs + m, updated = System.currentTimeMillis()))
    }

    private fun save(c: Conversation) { store.save(c); changed.value++ }

    private fun newId() = UUID.randomUUID().toString().take(8)
}

/** One chat for the whole app, so a reply keeps going when the screen is rebuilt or left. */
object Chat {
    @Volatile private var c: ChatController? = null
    fun get(ctx: Context): ChatController = c ?: synchronized(this) { c ?: ChatController(ctx).also { c = it } }
    fun busy() = c?.busy() == true
}
