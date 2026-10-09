package com.anima.npu

import org.json.JSONObject

/** The one tool Ling may call, in the OpenAI format llama.cpp's server passes to Ling's chat template. */
val DRAW_TOOL_JSON = """
{"type":"function","function":{"name":"draw_picture",
 "description":"Draw pictures on this phone. They are added to the picture queue and appear in the chat when finished.",
 "parameters":{"type":"object","properties":{
  "model":{"type":"string","enum":["anima","qwen"],"description":"anima: anime and illustration. qwen: photos, realistic scenes, text in pictures, and every edit."},
  "prompt":{"type":"string","description":"For anima: comma-separated tags (e.g. '1girl, solo, silver hair, rain, night'). For qwen: one full descriptive sentence or two."},
  "size":{"type":"string","enum":["square","portrait","landscape"]},
  "edit":{"type":"boolean","description":"true to change the picture the user attached in their last message (qwen only)"},
  "count":{"type":"integer","minimum":1,"maximum":4}},
 "required":["model","prompt"]}}}
""".trimIndent()

data class DrawArgs(val model: String, val prompt: String, val size: String, val edit: Boolean, val count: Int)

sealed class ToolCheck {
    data class Ok(val args: DrawArgs) : ToolCheck()
    data class Bad(val message: String) : ToolCheck()
}

/** Checks Ling's arguments; a Bad message goes back to Ling as the tool's answer so it can tell the user. */
fun checkDraw(argsJson: String, hasAttachedPicture: Boolean, qwenCanEdit: Boolean, installed: Set<String>): ToolCheck {
    val o = runCatching { JSONObject(argsJson) }.getOrNull() ?: return ToolCheck.Bad("The arguments were not valid JSON.")
    val model = o.optString("model")
    if (model !in setOf("anima", "qwen")) return ToolCheck.Bad("model must be \"anima\" or \"qwen\".")
    if (model !in installed) return ToolCheck.Bad("The $model model is not downloaded on this phone. Tell the user to download it on the Create tab.")
    val prompt = o.optString("prompt").trim()
    if (prompt.isEmpty()) return ToolCheck.Bad("prompt is empty.")
    val edit = o.optBoolean("edit", false)
    if (edit && model != "qwen") return ToolCheck.Bad("Only qwen can edit a picture.")
    if (edit && !hasAttachedPicture) return ToolCheck.Bad("There is no attached picture to edit. Ask the user to attach one.")
    if (edit && !qwenCanEdit) return ToolCheck.Bad("Editing needs the Standard Qwen download; the Small one cannot edit.")
    val size = o.optString("size", "square").takeIf { it in setOf("square", "portrait", "landscape") } ?: "square"
    return ToolCheck.Ok(DrawArgs(model, prompt, size, edit, o.optInt("count", 1).coerceIn(1, 4)))
}

fun systemPrompt(qwenName: String) = """
You are the assistant inside Anima, an app that draws pictures on this phone. Keep replies short and friendly.
When the user asks for a picture, call draw_picture. Choose the model yourself: "anima" for anime, manga and
illustration; "qwen" ($qwenName) for photos, realistic scenes, text inside pictures, and any edit of an attached
picture. If the user names a model, use that one. Write the prompt in that model's style: tags for anima, full
sentences for qwen. After the tool answers, tell the user in one sentence that the pictures are on their way.
You cannot see pictures; an attached picture is passed to the editing model as it is.
""".trimIndent()
