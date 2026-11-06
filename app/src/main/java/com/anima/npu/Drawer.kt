package com.anima.npu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** Draws one queued job with the right engine and saves the picture to the gallery. */
object Drawer {
    data class Result(val bitmap: Bitmap, val entry: Gallery.Entry, val ms: Long)

    /** [progress] gets (step, total): step 0 while reading the prompt. Throws with the engine's message on failure. */
    suspend fun draw(ctx: Context, j: Job, progress: (Int, Int) -> Unit): Result = withContext(Dispatchers.Default) {
        val err = when (j.model) {
            "klein" -> Engines.ensureKlein()
            "qwen" -> Engines.ensureQwen(j.qwenDrawing)
            "fibo" -> Engines.ensureFibo()
            "moss" -> Engines.ensureMoss()
            else -> Engines.ensureAnima()
        }
        if (err != null) error("Could not start: $err")
        val (w, h) = if (j.model in BIG_MODELS) 768 to 768 else pictureSize(if (j.model == "klein") 0 else j.tier, j.shape)
        val src = j.sourceFile?.let { BitmapFactory.decodeFile(File(Queue.dir(ctx), it).absolutePath) }
        val p = Progress { i, n -> progress(i, n) }
        val t0 = System.currentTimeMillis()
        val px = when (j.model) {
            "moss" -> {
                val (cond, unc) = MossReader.read(ctx, ModelStore.MOSS.modelDir(ctx), j.prompt)
                Engine.mossGenerate(cond, unc, j.seed, p)
            }
            "fibo" -> {
                val e = src?.let { guidePixels(it, w, h) } ?: error("Fibo only edits pictures: choose a picture to edit")
                Engine.fiboGenerate(QwenImageTokens.encodeFibo(j.prompt), j.seed, e, p)
            }
            "qwen" -> {
                val e = if (ModelStore.QWEN.installed(ctx) == "small") null else src?.let { guidePixels(it, w, h) }
                Engine.qwenGenerate(QwenImageTokens.encode(j.prompt, e != null), j.seed, w, h, e, p)
            }
            "klein" -> Engine.kleinGenerate(KleinTokens.encode(j.prompt), j.seed, 4, w, h, src?.let { guidePixels(it, w, h) }, p)
            else -> {
                val t = Tokens.encode(j.prompt)
                Engine.generate(t.qwen, t.t5, t.t5Weights, j.seed, j.steps, w, h, src?.let { guidePixels(it, w, h) },
                    j.guideStrength, j.guideEnd, null, p)
            }
        } ?: error(Engine.lastError())
        val ms = System.currentTimeMillis() - t0
        val bmp = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
        val entry = Gallery.save(ctx, bmp, j.prompt, j.seed, when (j.model) { "anima" -> j.steps; "qwen", "moss" -> 8; else -> 4 }, ms)
        Result(bmp, entry, ms)
    }
}
