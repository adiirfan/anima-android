package com.anima.npu

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import java.io.File

/**
 * The two tokenizers with ComfyUI's rules for Anima (comfy/sd1_clip.py SDTokenizer + text_encoders/anima.py):
 * "(text)" is emphasis x1.1, "(text:1.3)" sets it, "\(" and "\)" are literal brackets, every bracketed or
 * plain segment is tokenized on its own. Qwen ids carry no special tokens and no weights; T5 ids get each
 * segment's weight and one end token (1) at the very end.
 */
object Tokens {
    private lateinit var qwen: HuggingFaceTokenizer
    private lateinit var t5: HuggingFaceTokenizer

    class Result(val qwen: IntArray, val t5: IntArray, val t5Weights: FloatArray)

    fun init(modelDir: File) {
        if (::qwen.isInitialized) return
        qwen = build(File(modelDir, "qwen_tokenizer.json"))
        t5 = build(File(modelDir, "t5_tokenizer.json"))
    }

    private fun build(f: File): HuggingFaceTokenizer =
        HuggingFaceTokenizer.builder()
            .optTokenizerPath(f.toPath())
            .optAddSpecialTokens(false)
            .optTruncation(false)
            .optPadding(false)
            .build()

    fun encode(text: String): Result {
        val q = ArrayList<Int>()
        val t = ArrayList<Int>()
        val w = ArrayList<Float>()
        for ((segment, weight) in tokenWeights(escapeImportant(text), 1.0f)) {
            for (word in splitEmbeddings(unescapeImportant(segment))) {
                if (word.isEmpty()) continue
                qwen.encode(word, false, false).ids.forEach { q.add(it.toInt()) }
                t5.encode(word, false, false).ids.forEach { t.add(it.toInt()); w.add(weight) }
            }
        }
        t.add(1); w.add(1.0f)  // T5 end token
        return Result(q.toIntArray(), t.toIntArray(), w.toFloatArray())
    }

    private fun escapeImportant(s: String) = s.replace("\\)", "\u0000\u0001").replace("\\(", "\u0000\u0002")
    private fun unescapeImportant(s: String) = s.replace("\u0000\u0001", ")").replace("\u0000\u0002", "(")

    // ComfyUI tokenizes text before an "embedding:" name separately; with no embeddings folder the name is
    // then tokenized as plain text, but still as its own piece.
    private fun splitEmbeddings(s: String): List<String> {
        val parts = Regex("(?<=\\s)embedding:").split(s)
        return listOf(parts[0]) + parts.drop(1).map { "embedding:$it" }
    }

    private fun parseParentheses(s: String): List<String> {
        val result = ArrayList<String>()
        var cur = StringBuilder()
        var nesting = 0
        for (ch in s) {
            when (ch) {
                '(' -> {
                    if (nesting == 0) {
                        if (cur.isNotEmpty()) result.add(cur.toString())
                        cur = StringBuilder("(")
                    } else cur.append(ch)
                    nesting++
                }
                ')' -> {
                    nesting--
                    if (nesting == 0) {
                        result.add("$cur)")
                        cur = StringBuilder()
                    } else cur.append(ch)
                }
                else -> cur.append(ch)
            }
        }
        if (cur.isNotEmpty()) result.add(cur.toString())
        return result
    }

    private fun tokenWeights(s: String, current: Float): List<Pair<String, Float>> {
        val out = ArrayList<Pair<String, Float>>()
        for (item in parseParentheses(s)) {
            if (item.length >= 2 && item.first() == '(' && item.last() == ')') {
                var x = item.substring(1, item.length - 1)
                var weight = current * 1.1f
                val colon = x.lastIndexOf(':')
                if (colon > 0) {
                    x.substring(colon + 1).trim().toFloatOrNull()?.let { weight = it; x = x.substring(0, colon) }
                }
                out += tokenWeights(x, weight)
            } else {
                out += item to current
            }
        }
        return out
    }
}

/**
 * FLUX.2 klein's prompt: Qwen3's chat template around the plain text (ComfyUI's KleinTokenizer), no emphasis
 * syntax. The special tokens in the template are recognised by the tokenizer itself; checked against ComfyUI's
 * ids on 4 prompts (2026-10-07).
 */
object KleinTokens {
    private var qwen: HuggingFaceTokenizer? = null
    private const val TEMPLATE = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    fun init(modelDir: File) {
        if (qwen != null) return
        qwen = HuggingFaceTokenizer.builder()
            .optTokenizerPath(File(modelDir, "qwen_tokenizer.json").toPath())
            .optAddSpecialTokens(false)
            .optTruncation(false)
            .optPadding(false)
            .build()
    }

    fun encode(text: String): IntArray =
        qwen!!.encode(TEMPLATE.format(text), false, false).ids.map { it.toInt() }.toIntArray()
}

/**
 * Qwen-Image 2.1's prompt: ComfyUI's QwenImage21Tokenizer template (a system turn the drawing model never sees, then
 * the user turn). An edit puts the picture before the text as "<image1>" + the vision block, whose single
 * <|image_pad|> becomes the picture's 576 rows (24 x 24 at 768). Checked against ComfyUI's ids for a picture and an
 * edit prompt (2026-10-08).
 */
object QwenImageTokens {
    private var qwen: HuggingFaceTokenizer? = null
    private const val SYSTEM = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    private const val PAD = 151655   // <|image_pad|>
    const val PICTURE_ROWS = 576

    fun init(modelDir: File) {
        if (qwen != null) return
        qwen = HuggingFaceTokenizer.builder()
            .optTokenizerPath(File(modelDir, "qwen_tokenizer.json").toPath())
            .optAddSpecialTokens(false)
            .optTruncation(false)
            .optPadding(false)
            .build()
    }

    fun encode(text: String, edit: Boolean): IntArray {
        val picture = if (edit) "<image1><|vision_start|><|image_pad|><|vision_end|>" else ""
        val ids = qwen!!.encode("$SYSTEM<|im_start|>user\n$picture$text<|im_end|>\n<|im_start|>assistant\n", false, false).ids
        val out = ArrayList<Int>(ids.size + if (edit) PICTURE_ROWS else 0)
        for (id in ids) if (edit && id.toInt() == PAD) repeat(PICTURE_ROWS) { out.add(PAD) } else out.add(id.toInt())
        return out.toIntArray()
    }
}
