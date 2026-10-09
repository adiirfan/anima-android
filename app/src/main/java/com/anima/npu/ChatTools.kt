package com.anima.npu

import org.json.JSONObject

/** The one tool Ling may call, in the OpenAI format llama.cpp's server passes to Ling's chat template. */
val DRAW_TOOL_JSON = """
{"type":"function","function":{"name":"draw_picture",
 "description":"Draw pictures on this phone. They are added to the picture queue and appear in the chat when finished.",
 "parameters":{"type":"object","properties":{
  "model":{"type":"string","enum":["anima","qwen"],"description":"anima: anime and illustration. qwen: photos, realistic scenes, text in pictures, and every edit."},
  "prompt":{"type":"string","description":"For anima: comma-separated tags (e.g. '1girl, solo, silver hair, rain, night'). For qwen: one full descriptive sentence or two. For an edit: the change to make, as an instruction (e.g. 'Make it night time')."},
  "size":{"type":"string","enum":["square","portrait","landscape"]},
  "edit":{"type":"boolean","description":"true to change the newest picture in this chat (one you drew or one the user attached) instead of drawing a new one. Edits always use qwen."},
  "count":{"type":"integer","minimum":1,"maximum":4}},
 "required":["model","prompt"]}}}
""".trimIndent()

data class DrawArgs(val model: String, val prompt: String, val size: String, val edit: Boolean, val count: Int)

sealed class ToolCheck {
    data class Ok(val args: DrawArgs) : ToolCheck()
    data class Bad(val message: String) : ToolCheck()
}

/** Checks Ling's arguments; a Bad message goes back to Ling as the tool's answer so it can tell the user. */
fun checkDraw(argsJson: String, noPictureToEdit: String?, qwenCanEdit: Boolean, installed: Set<String>): ToolCheck {
    val o = runCatching { JSONObject(argsJson) }.getOrNull() ?: return ToolCheck.Bad("The arguments were not valid JSON.")
    val edit = o.optBoolean("edit", false)
    // only qwen edits; small models often ask anima for an edit anyway, so the edit wins over the model they named
    val model = if (edit) "qwen" else o.optString("model")
    if (model !in setOf("anima", "qwen")) return ToolCheck.Bad("model must be \"anima\" or \"qwen\".")
    if (model !in installed) return ToolCheck.Bad("The $model model is not downloaded on this phone. Tell the user to download it on the Create tab.")
    val prompt = o.optString("prompt").trim()
    if (prompt.isEmpty()) return ToolCheck.Bad("prompt is empty.")
    if (edit && noPictureToEdit != null) return ToolCheck.Bad(noPictureToEdit)
    if (edit && !qwenCanEdit) return ToolCheck.Bad("Editing needs the Standard Qwen download; the Small one cannot edit.")
    val size = o.optString("size", "square").takeIf { it in setOf("square", "portrait", "landscape") } ?: "square"
    return ToolCheck.Ok(DrawArgs(model, prompt, size, edit, o.optInt("count", 1).coerceIn(1, 4)))
}

fun systemPrompt(qwenName: String) = """
You are the assistant inside Anima, an app that draws pictures on this phone. Keep replies short and friendly.
Call draw_picture only when the user asks for a picture or a change to one; for greetings, questions and
conversation just answer in words, never drawing one as an example. Choose the model yourself: "anima" for anime, manga and
illustration; "qwen" ($qwenName) for photos, realistic scenes and text inside pictures. If the user names a model,
use that one. Write the prompt in that model's style: tags for anima, full sentences for qwen.
When the user wants to change a picture ("edit it", "make it night", "give her a hat"), call draw_picture with
edit true: it changes the newest picture in this chat, whether you drew it or the user attached it. The prompt is then
just the change, as an instruction. After the tool answers, tell the user in one sentence that the pictures are on
their way. You cannot see pictures; the picture is passed to the editing model as it is.
""".trimIndent()
