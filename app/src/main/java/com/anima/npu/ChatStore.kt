package com.anima.npu

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** One entry in a conversation. Pictures are queue jobs; they show once the job finishes. */
sealed class Msg {
    abstract val id: String
    data class User(override val id: String, val text: String, val imageFile: String?) : Msg()
    data class Assistant(override val id: String, val text: String, val thinking: String) : Msg()
    data class ToolCall(override val id: String, val callId: String, val name: String, val argsJson: String) : Msg()
    data class ToolResult(override val id: String, val callId: String, val resultJson: String) : Msg()
    data class Picture(override val id: String, val jobId: String) : Msg()

    fun toJson(): JSONObject = JSONObject().put("id", id).also { o ->
        when (this) {
            is User -> o.put("kind", "user").put("text", text).put("imageFile", imageFile ?: JSONObject.NULL)
            is Assistant -> o.put("kind", "assistant").put("text", text).put("thinking", thinking)
            is ToolCall -> o.put("kind", "tool_call").put("callId", callId).put("name", name).put("args", argsJson)
            is ToolResult -> o.put("kind", "tool_result").put("callId", callId).put("result", resultJson)
            is Picture -> o.put("kind", "picture").put("jobId", jobId)
        }
    }

    companion object {
        fun fromJson(o: JSONObject): Msg {
            val id = o.getString("id")
            return when (o.getString("kind")) {
                "user" -> User(id, o.getString("text"), if (o.isNull("imageFile")) null else o.getString("imageFile"))
                "assistant" -> Assistant(id, o.getString("text"), o.optString("thinking"))
                "tool_call" -> ToolCall(id, o.getString("callId"), o.getString("name"), o.getString("args"))
                "tool_result" -> ToolResult(id, o.getString("callId"), o.getString("result"))
                else -> Picture(id, o.getString("jobId"))
            }
        }
    }
}

data class Conversation(val id: String, val title: String, val updated: Long, val thinking: Boolean, val msgs: List<Msg>) {
    fun toJson(): JSONObject = JSONObject().put("id", id).put("title", title).put("updated", updated)
        .put("thinking", thinking).put("msgs", JSONArray().also { a -> msgs.forEach { a.put(it.toJson()) } })

    companion object {
        fun fromJson(o: JSONObject): Conversation {
            val a = o.getJSONArray("msgs")
            return Conversation(o.getString("id"), o.getString("title"), o.getLong("updated"), o.optBoolean("thinking"),
                List(a.length()) { Msg.fromJson(a.getJSONObject(it)) })
        }
    }
}

/** Conversations, one JSON file each (dir/<id>.json), written whole and atomically. */
class ChatStore(private val dir: File) {
    init { dir.mkdirs() }

    private fun file(id: String) = File(dir, "$id.json")

    @Synchronized fun list(): List<Conversation> =
        (dir.listFiles { f -> f.name.endsWith(".json") } ?: emptyArray())
            .mapNotNull { f -> runCatching { Conversation.fromJson(JSONObject(f.readText())) }.getOrNull() }
            .sortedByDescending { it.updated }

    @Synchronized fun get(id: String): Conversation? =
        file(id).takeIf { it.isFile }?.let { f -> runCatching { Conversation.fromJson(JSONObject(f.readText())) }.getOrNull() }

    @Synchronized fun save(c: Conversation) {
        val tmp = File(dir, "${c.id}.json.tmp")
        tmp.writeText(c.toJson().toString())
        if (!tmp.renameTo(file(c.id))) { file(c.id).delete(); tmp.renameTo(file(c.id)) }
    }

    @Synchronized fun delete(id: String) {
        file(id).delete()
        // attached pictures are saved beside the conversation as <id>-<n>.png
        dir.listFiles { f -> f.name.startsWith("$id-") }?.forEach { it.delete() }
    }

    @Synchronized fun rename(id: String, title: String) { get(id)?.let { save(it.copy(title = title)) } }

    fun dir() = dir
}
