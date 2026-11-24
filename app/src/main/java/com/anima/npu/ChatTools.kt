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

/** What the user's own words fixed for a picture: the model ("anima"/"qwen"), size (index into SIZE_TIERS), Anima steps. */
data class Named(val model: String?, val tier: Int?, val steps: Int?)

private val BIG_ASK = Regex("""\b768\b|\b(large|big|bigger|larger|full|high)[- ]?(size|sized|picture|image|version|res|resolution)\b|\bhd\b""",
    RegexOption.IGNORE_CASE)
private val SMALL_ASK = Regex("""\b512\b|\bsmall[- ]?(size|sized|picture|image|version)\b""", RegexOption.IGNORE_CASE)
private val STEPS_ASK = Regex("""\b(\d{1,3})\s*steps?\b""", RegexOption.IGNORE_CASE)

/**
 * Reads the model, size and steps from what the user typed: "Qwen", "large picture", "768", "HD", "12 steps".
 * Read from their words rather than left to the small chat model, which picked anything it liked.
 */
fun namedSettings(text: String): Named = Named(
    model = when {
        Regex("""\b(qwen|noct\s?q)\b""", RegexOption.IGNORE_CASE).containsMatchIn(text) -> "qwen"
        Regex("""\banima\b""", RegexOption.IGNORE_CASE).containsMatchIn(text) -> "anima"
        else -> null
    },
    tier = if (BIG_ASK.containsMatchIn(text)) 1 else if (SMALL_ASK.containsMatchIn(text)) 0 else null,
    steps = STEPS_ASK.find(text)?.groupValues?.get(1)?.toIntOrNull()?.takeIf { it > 0 }?.coerceAtMost(30),
)

private val QUESTION = Regex("""^\s*(how|what|why|which|who|where|when|is|are|do|does)\b""", RegexOption.IGNORE_CASE)
private val PICTURE_ASKS = listOf(
    Regex("""^\s*(please\s+)?((can|could|would)\s+you\s+)?(please\s+)?(draw|paint|sketch|illustrate)\b""", RegexOption.IGNORE_CASE),
    Regex("""\b(make|create|generate|give me|show me|draw)\b.{0,40}\b(picture|image|drawing|illustration|photo|pic|wallpaper)s?\b""", RegexOption.IGNORE_CASE),
    Regex("""\b(picture|image|photo|drawing|illustration)s?\s+of\b""", RegexOption.IGNORE_CASE),
)

/** The user asked for a picture ("draw a cat", "make me a picture of…"); questions about pictures are not asks. */
fun wantsPicture(text: String): Boolean = !QUESTION.containsMatchIn(text) && PICTURE_ASKS.any { it.containsMatchIn(text) }

private val ASK_LEAD = Regex("""^\s*(please\s+)?((can|could|would)\s+you\s+)?(please\s+)?(draw|paint|sketch|illustrate|make|create|generate|give|show)\s+""" +
    """(me\s+)?((a|an|the|some)\s+)?((picture|image|drawing|illustration|photo|pic)s?\s+of\s+)?((a|an|the|some)\s+)?""", RegexOption.IGNORE_CASE)
private val ASK_TAIL = Regex("""[\s,]*(please)?[\s?.!]*$""", RegexOption.IGNORE_CASE)

/** The user's words as the picture's description, without the asking ("draw a…", "make me a picture of…"). */
fun pictureWords(text: String): String {
    val w = text.trim().replaceFirst(ASK_LEAD, "").replace(ASK_TAIL, "").trim()
    return if (w.equals("something", true) || w.equals("anything", true)) "" else w
}

/** What to do with a picture request: draw it now, ask on a card for what the user did not name, or refuse. */
sealed class DrawPlan {
    data class Queue(val model: String, val tier: Int, val steps: Int) : DrawPlan()
    /** [model], [tier] and [steps] start ticked; the ask* rows are shown (size and steps only matter to Anima). */
    data class Ask(val model: String, val askModel: Boolean, val askSize: Boolean, val askSteps: Boolean, val tier: Int, val steps: Int) : DrawPlan()
    data class Bad(val message: String) : DrawPlan()
}

/**
 * [suggested] is the model the chat picked, [installed] the picture models on this phone (only Anima and Qwen draw
 * from chat). Edits always draw at once with Qwen. Otherwise anything the user did not name is asked on a card,
 * starting from the chat's pick (if it is downloaded), 512 and 4 steps.
 */
fun planDraw(suggested: String, edit: Boolean, named: Named, installed: Set<String>): DrawPlan {
    if (edit) return if ("qwen" in installed) DrawPlan.Queue("qwen", 0, 8)
        else DrawPlan.Bad("Edits use Qwen, which is not downloaded on this phone. Tell the user to download it on the Create tab.")
    val have = listOf("anima", "qwen").filter { it in installed }
    if (have.isEmpty()) return DrawPlan.Bad("No picture model is downloaded on this phone. Tell the user to download Anima or Qwen on the Create tab.")
    named.model?.let { m -> if (m !in have) return DrawPlan.Bad("The $m model is not downloaded on this phone. Tell the user to download it on the Create tab.") }
    val model = named.model ?: suggested.takeIf { it in have } ?: have.first()
    val askModel = named.model == null && have.size > 1
    val askSize = named.tier == null
    val askSteps = named.steps == null
    val tier = named.tier ?: 0
    val steps = named.steps ?: 4
    if (!askModel && (model == "qwen" || (!askSize && !askSteps))) return DrawPlan.Queue(model, if (model == "qwen") 0 else tier, if (model == "qwen") 8 else steps)
    return DrawPlan.Ask(model, askModel, askSize, askSteps, tier, steps)
}
