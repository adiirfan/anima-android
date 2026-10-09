package com.anima.npu

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/** One piece of a streamed reply. */
sealed class Event {
    data class Text(val delta: String) : Event()
    data class Thinking(val delta: String) : Event()
    /** A piece of tool call [index]; [id] and [name] come with the first piece only. */
    data class Tool(val index: Int, val id: String?, val name: String?, val argsDelta: String) : Event()
    object Done : Event()
}

/** Turns the server's streamed lines ("data: {...}", OpenAI format) into events; other lines give null. */
class SseAccumulator {
    fun feed(line: String): Event? {
        if (!line.startsWith("data:")) return null
        val body = line.removePrefix("data:").trim()
        if (body == "[DONE]") return Event.Done
        val o = runCatching { JSONObject(body) }.getOrNull() ?: return null
        val d = o.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("delta") ?: return null
        fun str(k: String) = if (d.has(k) && !d.isNull(k)) d.getString(k) else null
        d.optJSONArray("tool_calls")?.optJSONObject(0)?.let { t ->
            val f = t.optJSONObject("function")
            return Event.Tool(t.optInt("index"), if (t.has("id") && !t.isNull("id")) t.getString("id") else null,
                f?.let { if (it.has("name") && !it.isNull("name")) it.getString("name") else null },
                f?.optString("arguments", "") ?: "")
        }
        str("reasoning_content")?.takeIf { it.isNotEmpty() }?.let { return Event.Thinking(it) }
        str("content")?.takeIf { it.isNotEmpty() }?.let { return Event.Text(it) }
        return null
    }
}

/** Talks to Ling's server on this phone (OpenAI-style chat, streamed). */
class ChatClient(private val port: Int = LocalLing.PORT) {
    suspend fun stream(messages: JSONArray, tools: JSONArray, thinking: Boolean, cancel: () -> Boolean, on: (Event) -> Unit) =
        withContext(Dispatchers.IO) {
            val body = JSONObject().put("model", "ling").put("stream", true).put("messages", messages)
                .put("tools", tools).put("chat_template_kwargs", JSONObject().put("enable_thinking", thinking))
            val c = URL("http://127.0.0.1:$port/v1/chat/completions").openConnection() as HttpURLConnection
            c.connectTimeout = 10_000
            c.readTimeout = 300_000
            c.requestMethod = "POST"
            c.doOutput = true
            c.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            c.setRequestProperty("Accept", "text/event-stream")
            try {
                c.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
                if (c.responseCode != 200) {
                    val err = (c.errorStream ?: c.inputStream).bufferedReader().readText().take(300)
                    error("Ling answered ${c.responseCode}: $err")
                }
                val p = SseAccumulator()
                c.inputStream.bufferedReader(Charsets.UTF_8).use { r ->
                    while (true) {
                        if (cancel()) break
                        val line = r.readLine() ?: break
                        val e = p.feed(line) ?: continue
                        on(e)
                        if (e == Event.Done) break
                    }
                }
            } finally {
                c.disconnect()
            }
        }
}
