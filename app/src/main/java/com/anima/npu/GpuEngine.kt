package com.anima.npu

import android.content.Context
import android.util.Log
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Drawing on the graphics chip instead of the AI chip: the app's own engine (built from ~/ugpu as libugpu.so, a
 * program like libllama-server.so), standard OpenCL only, so it can run on phones without a Snapdragon AI chip.
 * Slower than the AI chip (MOSS on the Fold: ~25 s a step against ~9 s). Each run is its own process: a problem
 * there ends that run, not the app, and all its memory is free again afterwards.
 *
 * MOSS: the reader's rows go in, the engine draws with the 8-step weights (moss_q8.gguf, streamed one block at a
 * time because it is bigger than the phone's free memory), then the small Flux 2 decoder (tae_f2.gguf) makes the
 * picture. Weights live in models/moss/gpu/.
 */
object GpuEngine {
    private const val LIB = "libugpu.so"
    private const val TAG = "ugpu"

    private var state: MutableState<Boolean>? = null
    private fun st(ctx: Context) = state ?: mutableStateOf(
        ctx.getSharedPreferences("anima", Context.MODE_PRIVATE).getBoolean("gpuEngine", false)).also { state = it }

    /** Settings: "Draw MOSS with the graphics chip" (always on for phones without a supported AI chip). */
    fun forced(ctx: Context) = st(ctx).value
    fun setForced(ctx: Context, on: Boolean) {
        st(ctx).value = on
        ctx.getSharedPreferences("anima", Context.MODE_PRIVATE).edit().putBoolean("gpuEngine", on).apply()
    }

    /** models/moss itself when the graphics-chip build was downloaded (phones without the AI chip), else models/moss/gpu
     *  (copied there by hand next to the AI-chip files, for testing). */
    fun weightsDir(ctx: Context): File {
        val m = ModelStore.MOSS.modelDir(ctx)
        return if (File(m, "moss_q8.gguf").isFile) m else File(m, "gpu")
    }
    private fun program(ctx: Context) = File(ctx.applicationInfo.nativeLibraryDir, LIB)

    /** The engine program and MOSS's graphics-chip weights are both on the phone. */
    fun mossReady(ctx: Context) = program(ctx).isFile &&
        File(weightsDir(ctx), "moss_q8.gguf").isFile && File(weightsDir(ctx), "tae_f2.gguf").isFile

    /** Draw MOSS on the graphics chip: on request, or because the phone has no AI chip the app supports. */
    fun useForMoss(ctx: Context) = (forced(ctx) || !Chip.supported) && mossReady(ctx)

    /** The engine's code for the chip, copied from the app's assets next to the work folders (it is read at start). */
    private fun home(ctx: Context): File {
        val d = File(ctx.filesDir, "ugpu").apply { mkdirs() }
        for (n in listOf("kernels.cl", "tae.cl")) {
            val bytes = ctx.assets.open("ugpu/$n").use { it.readBytes() }
            val f = File(d, n)
            if (!f.isFile || f.length() != bytes.size.toLong() || !f.readBytes().contentEquals(bytes)) f.writeBytes(bytes)
        }
        return d
    }

    /** Runs the engine once; [line] sees every line it prints. Throws with its last lines when it fails. */
    private fun run(ctx: Context, args: List<String>, env: Map<String, String>, line: (String) -> Unit = {}) =
        runProgram(ctx, LIB, args, env, line)

    /** Runs [lib] (a program in the app's library folder) once, in the engine's folder. */
    private fun runProgram(ctx: Context, lib: String, args: List<String>, env: Map<String, String>, line: (String) -> Unit = {}) {
        val home = home(ctx)
        val libs = ctx.applicationInfo.nativeLibraryDir
        val pb = ProcessBuilder(listOf(File(libs, lib).absolutePath) + args).directory(home).redirectErrorStream(true)
        pb.environment()["LD_LIBRARY_PATH"] = "$libs:/system/lib64:/vendor/lib64:/system/vendor/lib64"
        pb.environment().putAll(env)
        val proc = pb.start()
        val last = ArrayDeque<String>()
        try {
            proc.inputStream.bufferedReader().forEachLine { l ->
                if (l.isBlank()) return@forEachLine
                Log.i(TAG, l.take(300))
                last.addLast(l.take(200)); if (last.size > 4) last.removeFirst()
                line(l)
            }
            val code = proc.waitFor()
            if (code != 0) error("$lib stopped (code $code): ${last.joinToString(" | ")}")
        } finally {
            if (proc.isAlive) proc.destroy()
        }
    }

    private fun writeF32(f: File, a: FloatArray) {
        val b = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        b.asFloatBuffer().put(a)
        f.writeBytes(b.array())
    }

    private fun readF32(f: File): FloatArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(b.remaining()).also { b.get(it) }
    }

    /** 768 x 768 ARGB pixels. [cond]: the reader's rows (rows x 4096). Progress counts the 8 steps. */
    fun mossGenerate(ctx: Context, cond: FloatArray, seed: Long, progress: Progress): IntArray {
        val S = 768; val h = S / 16; val w = S / 16; val C = 128; val ctxW = 4096
        val wdir = weightsDir(ctx)
        val sigmas = File(ModelStore.MOSS.modelDir(ctx), "sigmas.txt").takeIf { it.isFile }?.readText()
            ?.split(Regex("\\s+"))?.mapNotNull { it.toFloatOrNull() }?.filter { it > 0f && it <= 1f }
            ?.takeIf { it.isNotEmpty() } ?: listOf(1.0f, 0.968f, 0.926f, 0.871f, 0.791f, 0.671f, 0.463f, 0.024f)
        val nc = cond.size / ctxW
        val work = File(ctx.cacheDir, "ugpu_moss").apply { deleteRecursively(); mkdirs() }
        File(work, "info.txt").writeText("$nc $S $S\n")
        writeF32(File(work, "ctx.f32"), cond)
        val rnd = java.util.Random(seed)
        writeF32(File(work, "noise.f32"), FloatArray(h * w * C) { rnd.nextGaussian().toFloat() })
        writeF32(File(work, "sigmas.f32"), sigmas.toFloatArray())
        val total = sigmas.size
        progress.onStep(0, total)
        val home = home(ctx)
        val step = Regex("^draw step (\\d+):")
        run(ctx, listOf(File(wdir, "moss_q8.gguf").absolutePath, work.absolutePath),
            mapOf("DRAW" to "1", "STREAM" to "1", "TUNEFILE" to File(home, "moss.tune").absolutePath)) { l ->
            step.find(l)?.let { progress.onStep(it.groupValues[1].toInt() + 1, total) }
        }
        val out = decodeFlux2(ctx, File(work, "draw.f32"), h, w, wdir)
        work.deleteRecursively()
        return out
    }

    /** The drawing's latent (h x w rows of 128: each 2 x 2 patch packed into 4 channels) -> ARGB pixels, through the
     *  small Flux 2 decoder. Unpacks (y, x, c32, i, j) -> (2y + i, 2x + j, c32) as ComfyUI's DecoderFlux2. */
    private fun decodeFlux2(ctx: Context, drawn: File, h: Int, w: Int, wdir: File): IntArray {
        val C = 128
        val z = readF32(drawn)
        if (z.size != h * w * C) error("graphics-chip engine gave ${z.size} numbers, not ${h * w * C}")
        val lat = FloatArray(z.size)
        for (y in 0 until h) for (x in 0 until w) for (c in 0 until 32) for (i in 0 until 2) for (j in 0 until 2)
            lat[((2 * y + i) * (2 * w) + 2 * x + j) * 32 + c] = z[((y * w + x) * 32 + c) * 4 + i * 2 + j]
        val dec = File(drawn.parentFile, "dec").apply { deleteRecursively(); mkdirs() }
        writeF32(File(dec, "lat.f32"), lat)
        File(dec, "info.txt").writeText("${2 * h} ${2 * w}\n")
        run(ctx, listOf(File(wdir, "tae_f2.gguf").absolutePath, dec.absolutePath),
            mapOf("TUNEFILE" to File(home(ctx), "tae_f2.tune").absolutePath))
        val rgb = readF32(File(dec, "gpu_rgb.f32"))
        val H = h * 16; val W = w * 16
        if (rgb.size != H * W * 3) error("picture decoder gave ${rgb.size} numbers, not ${H * W * 3}")
        fun px(v: Float) = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        return IntArray(H * W) { k ->
            (0xFF shl 24) or (px(rgb[3 * k]) shl 16) or (px(rgb[3 * k + 1]) shl 8) or px(rgb[3 * k + 2])
        }
    }

    // ---- Fibo 2.0 edit-turbo ----
    private const val FIBO_FILES = "fibo_q8.gguf tae_f2.gguf taee_f2.gguf fibo_reader.gguf fibo_reader_mmproj.gguf"

    /** models/fibo itself when the graphics-chip build was downloaded, else models/fibo/gpu (copied by hand, for testing). */
    fun fiboDir(ctx: Context): File {
        val m = ModelStore.FIBO.modelDir(ctx)
        return if (File(m, "fibo_q8.gguf").isFile) m else File(m, "gpu")
    }

    fun fiboReady(ctx: Context) = program(ctx).isFile && File(ctx.applicationInfo.nativeLibraryDir, "libfibotaps.so").isFile &&
        FIBO_FILES.split(" ").all { File(fiboDir(ctx), it).isFile }

    /** Fibo on the graphics chip: on request (same switch as MOSS), or on phones without a supported AI chip. */
    fun useForFibo(ctx: Context) = (forced(ctx) || !Chip.supported) && fiboReady(ctx)

    /**
     * Edits [edit] (768 x 768 ARGB) as [instruction] says. The reader (Qwen3-VL-4B, 8-bit, on the processor: its own
     * program, libfibotaps.so) reads the instruction with the picture and hands Fibo 15 of its inner layers; the
     * small Flux 2 encoder turns the picture into the edit's starting rows; the engine draws the 4 steps; the small
     * decoder makes the picture. Progress: 0 while reading, then the 4 steps.
     */
    fun fiboGenerate(ctx: Context, instruction: String, seed: Long, edit: IntArray, progress: Progress): IntArray {
        val S = 768; val h = S / 16; val w = S / 16; val C = 128
        if (edit.size != S * S) error("Fibo edits 768 x 768 pictures")
        val wdir = fiboDir(ctx)
        val sigmas = File(wdir, "sigmas.txt").takeIf { it.isFile }?.readText()
            ?.split(Regex("\\s+"))?.mapNotNull { it.toFloatOrNull() }?.filter { it > 0f && it <= 1f }
            ?.takeIf { it.isNotEmpty() } ?: listOf(1.0f, 0.904531f, 0.759511f, 0.512844f)
        val total = sigmas.size
        progress.onStep(0, total)
        val work = File(ctx.cacheDir, "ugpu_fibo").apply { deleteRecursively(); mkdirs() }
        val home = home(ctx)
        // 1. the reader: the picture as raw RGB rows, the instruction as a file (any characters survive)
        val rgb = ByteArray(S * S * 3)
        for (i in edit.indices) { val c = edit[i]; rgb[3 * i] = (c shr 16).toByte(); rgb[3 * i + 1] = (c shr 8).toByte(); rgb[3 * i + 2] = c.toByte() }
        File(work, "pic.rgb").writeBytes(rgb)
        File(work, "instr.txt").writeText(instruction.trim())
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8).toString()
        runProgram(ctx, "libfibotaps.so", listOf(File(wdir, "fibo_reader.gguf").absolutePath,
            File(wdir, "fibo_reader_mmproj.gguf").absolutePath, File(work, "pic.rgb").absolutePath, "$S", "$S",
            File(work, "instr.txt").absolutePath, work.absolutePath, threads), emptyMap())
        // 2. the picture's starting rows: the small encoder (picture in [0, 1], a zero 4th channel), packed 2 x 2
        val enc = File(work, "enc").apply { mkdirs() }
        val pic = FloatArray(S * S * 4)
        for (i in edit.indices) { val c = edit[i]
            pic[4 * i] = ((c shr 16) and 255) / 255f; pic[4 * i + 1] = ((c shr 8) and 255) / 255f; pic[4 * i + 2] = (c and 255) / 255f }
        writeF32(File(enc, "lat.f32"), pic)
        File(enc, "info.txt").writeText("$S $S\n")
        run(ctx, listOf(File(wdir, "taee_f2.gguf").absolutePath, enc.absolutePath),
            mapOf("TUNEFILE" to File(home, "taee_f2.tune").absolutePath))
        val e = readF32(File(enc, "gpu_rgb.f32"))   // (2h, 2w, 32)
        if (e.size != 4 * h * w * 32) error("picture encoder gave ${e.size} numbers, not ${4 * h * w * 32}")
        val ref = FloatArray(h * w * C)
        for (y in 0 until h) for (x in 0 until w) for (c in 0 until 32) for (i in 0 until 2) for (j in 0 until 2)
            ref[((y * w + x) * 32 + c) * 4 + i * 2 + j] = e[((2 * y + i) * (2 * w) + 2 * x + j) * 32 + c]
        writeF32(File(work, "ref.f32"), ref)
        // 3. the drawing
        val rnd = java.util.Random(seed)
        writeF32(File(work, "noise.f32"), FloatArray(h * w * C) { rnd.nextGaussian().toFloat() })
        writeF32(File(work, "sigmas.f32"), sigmas.toFloatArray())
        val step = Regex("^draw step (\\d+):")
        run(ctx, listOf(File(wdir, "fibo_q8.gguf").absolutePath, work.absolutePath),
            mapOf("DRAW" to "1", "STREAM" to "1", "TUNEFILE" to File(home, "fibo.tune").absolutePath)) { l ->
            step.find(l)?.let { progress.onStep(it.groupValues[1].toInt() + 1, total) }
        }
        val out = decodeFlux2(ctx, File(work, "draw.f32"), h, w, wdir)
        work.deleteRecursively()
        return out
    }
}
