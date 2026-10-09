package com.anima.npu

import java.io.File

fun interface Progress {
    fun onStep(step: Int, total: Int)
}

object Engine {
    init {
        System.loadLibrary("anima")
    }

    /** Files the app needs in its models folder (copied over the cable by push_models.sh). */
    val MODEL_FILES = listOf(
        "text.bin", "part0.bin", "part1.bin", "part2.bin", "part3.bin", "dec.bin", "xlsr.bin", "guide.bin",
        "qwen_embed_q8.raw", "t5_embed_f16.raw", "qwen_tokenizer.json", "t5_tokenizer.json",
    )

    fun missing(modelDir: File): List<String> = MODEL_FILES.filter { !File(modelDir, it).isFile }

    /** Opens the neural unit and maps the lookup tables; returns an error message or null. */
    external fun init(modelDir: String, libDir: String): String?

    /**
     * width*height ARGB pixels, or null (then see [lastError]). [guide]: optional width*height ARGB picture the
     * layout follows, for the first [guideEnd] of the steps at [guideStrength]. [noise] is only for the self-test.
     */
    external fun generate(
        qwenIds: IntArray, t5Ids: IntArray, t5Weights: FloatArray, seed: Long, steps: Int, width: Int, height: Int,
        guide: IntArray?, guideStrength: Float, guideEnd: Float, noise: FloatArray?, progress: Progress,
    ): IntArray?

    /** width x height -> 4x on the neural unit (XLSR, tiled), or null. */
    external fun upscale(pixels: IntArray, width: Int, height: Int): IntArray?

    /** Picture sizes ("WxH") the installed model was built for; call after [init]. */
    external fun sizes(): Array<String>

    /** True when the installed model can follow a guide picture. */
    external fun canGuide(): Boolean

    /** [loading ms, prompt ms, drawing ms, decode (or upscale) ms] of the last call. */
    external fun lastTimes(): DoubleArray
    external fun lastLatent(): FloatArray
    external fun lastError(): String

    /** 0 fastest (everything loaded), 1 normal, 2 low, 3 lowest memory (see Pipeline::setMemoryMode). */
    external fun setMemoryMode(mode: Int)

    /** Loading options (bits): 1 map the file, 2 init acceleration, 4 parallel weight copy. */
    external fun setLoadOptions(bits: Int)

    // ---- FLUX.2 klein (its own model folder, models/klein) ----

    /** Opens the Klein files (word table, sizes); returns an error message or null. */
    external fun kleinInit(modelDir: String, libDir: String): String?

    /** width*height ARGB pixels, or null (then see [lastError]). [edit]: optional width*height ARGB picture to change. */
    external fun kleinGenerate(ids: IntArray, seed: Long, steps: Int, width: Int, height: Int, edit: IntArray?, progress: Progress): IntArray?

    /** [loading ms, prompt ms, drawing ms, decode ms] of the last Klein picture. */
    external fun kleinTimes(): DoubleArray
    external fun kleinSetMemoryMode(mode: Int)

    // ---- Qwen-Image 2.1 turbo (its own model folder, models/qwen), 768x768 ----

    /** Opens the Qwen-Image files (word table); returns an error message or null. */
    /** drawDir: folder of the drawing model files (sp<k>.bin) to use instead of modelDir's, e.g. Noct Q; "" = modelDir. */
    external fun qwenInit(modelDir: String, libDir: String, drawDir: String): String?

    /**
     * 768*768 ARGB pixels, or null (then see [lastError]). [edit]: optional 768*768 ARGB picture to change, with
     * [ids] from [QwenImageTokens.encode] (edit = true). progress(i, 4) as drawing step i starts, 0 while reading.
     */
    external fun qwenGenerate(ids: IntArray, seed: Long, width: Int, height: Int, edit: IntArray?, progress: Progress): IntArray?

    /** [picture ms (edits), prompt ms, drawing ms, decode ms] of the last Qwen-Image picture. */
    external fun qwenTimes(): DoubleArray

    /** Total time spent running graphs on the neural unit so far, in ns (for the usage monitor). */
    external fun npuBusyNs(): Long

    /** Frees every piece on the neural unit (called when the app goes to the background). */
    external fun release()
}
