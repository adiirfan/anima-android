package com.anima.npu

import org.json.JSONObject

enum class JobState { WAITING, DRAWING, DONE, FAILED, CANCELLED }

/**
 * One picture to draw. [sourceFile]: the guide picture (Anima) or the picture to edit (Klein/Qwen), saved in the
 * queue folder. [origin]: "create" or "chat:<conversation id>". [qwenDrawing]: "qwen" or "noctq".
 */
data class Job(
    val id: String, val created: Long, val model: String, val qwenDrawing: String, val prompt: String,
    val seed: Long, val steps: Int, val tier: Int, val shape: Int, val guideStrength: Float, val guideEnd: Float,
    val sourceFile: String?, val origin: String,
    val state: JobState = JobState.WAITING, val resultUri: String? = null, val error: String? = null,
    val ms: Long = 0, val width: Int = 0, val height: Int = 0,
    val tries: Int = 0,  // times the app was closed while drawing it
) {
    fun toJson(): JSONObject = JSONObject().put("id", id).put("created", created).put("model", model)
        .put("qwenDrawing", qwenDrawing).put("prompt", prompt).put("seed", seed).put("steps", steps).put("tier", tier)
        .put("shape", shape).put("guideStrength", guideStrength.toDouble()).put("guideEnd", guideEnd.toDouble())
        .put("sourceFile", sourceFile ?: JSONObject.NULL).put("origin", origin).put("state", state.name)
        .put("resultUri", resultUri ?: JSONObject.NULL).put("error", error ?: JSONObject.NULL)
        .put("ms", ms).put("width", width).put("height", height).put("tries", tries)

    companion object {
        private fun JSONObject.str(k: String): String? = if (!has(k) || isNull(k)) null else getString(k)
        fun fromJson(o: JSONObject) = Job(
            o.getString("id"), o.getLong("created"), o.getString("model"), o.optString("qwenDrawing", "qwen"),
            o.getString("prompt"), o.getLong("seed"), o.getInt("steps"), o.getInt("tier"), o.getInt("shape"),
            o.getDouble("guideStrength").toFloat(), o.getDouble("guideEnd").toFloat(), o.str("sourceFile"),
            o.getString("origin"), JobState.valueOf(o.getString("state")), o.str("resultUri"), o.str("error"),
            o.optLong("ms"), o.optInt("width"), o.optInt("height"), o.optInt("tries"),
        )
    }
}
