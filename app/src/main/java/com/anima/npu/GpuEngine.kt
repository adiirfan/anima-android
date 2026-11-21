package com.anima.npu

import android.content.Context
import android.util.Log
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

    /** MOSS's graphics-chip build: models/moss/gpu (the models page downloads it there, next to the AI-chip build). */
    fun weightsDir(ctx: Context): File = ModelStore.MOSS.gpu!!.modelDir(ctx)
    private fun program(ctx: Context) = File(ctx.applicationInfo.nativeLibraryDir, LIB)

    /** The engine program and MOSS's graphics-chip weights are both on the phone. */
    fun mossReady(ctx: Context) = program(ctx).isFile &&
        File(weightsDir(ctx), "moss_q8.gguf").isFile && File(weightsDir(ctx), "tae_f2.gguf").isFile

    /** Draw MOSS on the graphics chip: picked on the models page, or the phone has no AI chip the app supports. */
    fun useForMoss(ctx: Context) = ModelStore.MOSS.runsOnGpu(ctx) && mossReady(ctx)

    /** The engine's code for the chip, copied from the app's assets next to the work folders (it is read at start). */
    private fun home(ctx: Context): File {
        val d = File(ctx.filesDir, "ugpu").apply { mkdirs() }
        for (n in listOf("kernels.cl", "tae.cl")) {
            val bytes = ctx.assets.open("ugpu/$n").use { it.readBytes() }
            val f = File(d, n)
            if (!f.isFile || f.length() != bytes.size.toLong() || !f.readBytes().contentEquals(bytes)) {
                f.writeBytes(bytes)
                File(d, "kcache").deleteRecursively()   // programs built from the old code are never used again
            }
        }
        File(d, "kcache").mkdirs()
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
        pb.environment()["KCACHE"] = File(home, "kcache").absolutePath   // built chip programs kept between runs
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
        val sigmas = File(weightsDir(ctx), "sigmas.txt").takeIf { it.isFile }?.readText()
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
    fun fiboDir(ctx: Context): File = ModelStore.FIBO.gpu!!.modelDir(ctx)

    fun fiboReady(ctx: Context) = program(ctx).isFile && File(ctx.applicationInfo.nativeLibraryDir, "libfibotaps.so").isFile &&
        FIBO_FILES.split(" ").all { File(fiboDir(ctx), it).isFile }

    /** Fibo on the graphics chip: on request (same switch as MOSS), or on phones without a supported AI chip. */
    fun useForFibo(ctx: Context) = ModelStore.FIBO.runsOnGpu(ctx) && fiboReady(ctx)

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

    // ---- Anima ----
    private const val ANIMA_FILES = "anima_q8.gguf anima_q8.cen anima_reader.gguf anima_adapter.gguf tae_w21.gguf qwen_tokenizer.json t5_tokenizer.json"

    fun animaDir(ctx: Context): File = ModelStore.ANIMA.gpu!!.modelDir(ctx)

    fun animaReady(ctx: Context) = program(ctx).isFile && File(ctx.applicationInfo.nativeLibraryDir, "libtexttaps.so").isFile &&
        ANIMA_FILES.split(" ").all { File(animaDir(ctx), it).isFile }

    /** Anima on the graphics chip: picked on the models page, or the phone has no AI-chip build of it. */
    fun useForAnima(ctx: Context) = ModelStore.ANIMA.runsOnGpu(ctx) && animaReady(ctx)

    /**
     * [width] x [height] ARGB pixels. The reader (Qwen3-0.6B, half precision) and Anima's prompt adapter run on the
     * processor (libtexttaps.so, mode anima), the engine draws [steps] steps (8-bit, centred), the small Wan 2.1
     * decoder makes the picture. Progress: 0 while reading, then the steps.
     */
    fun animaGenerate(ctx: Context, qwen: IntArray, t5: IntArray, t5w: FloatArray, seed: Long, steps: Int, width: Int,
                      height: Int, progress: Progress): IntArray {
        val hp = height / 16; val wp = width / 16
        val wdir = animaDir(ctx)
        progress.onStep(0, steps)
        val work = File(ctx.cacheDir, "ugpu_anima").apply { deleteRecursively(); mkdirs() }
        val home = home(ctx)
        fun ints(f: File, a: IntArray) {
            val b = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN); b.asIntBuffer().put(a); f.writeBytes(b.array())
        }
        ints(File(work, "qwen.i32"), qwen); ints(File(work, "t5.i32"), t5)
        writeF32(File(work, "t5w.f32"), if (t5w.size == t5.size) t5w else FloatArray(t5.size) { 1f })
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8).toString()
        runProgram(ctx, "libtexttaps.so", listOf(File(wdir, "anima_reader.gguf").absolutePath, File(work, "qwen.i32").absolutePath,
            work.absolutePath, threads, "anima", File(wdir, "anima_adapter.gguf").absolutePath,
            File(work, "t5.i32").absolutePath, File(work, "t5w.f32").absolutePath), emptyMap())
        // the noise latent (16, 2hp, 2wp) as patches: 'c (h m) (w n) -> (h w) (c m n)' with a zero 17th channel
        val rnd = java.util.Random(seed)
        val lat = FloatArray(16 * 4 * hp * wp) { rnd.nextGaussian().toFloat() }
        val x = FloatArray(hp * wp * 68)
        for (c in 0 until 16) for (y in 0 until 2 * hp) for (xx in 0 until 2 * wp)
            x[((y / 2) * wp + xx / 2) * 68 + c * 4 + (y % 2) * 2 + xx % 2] = lat[(c * 2 * hp + y) * 2 * wp + xx]
        writeF32(File(work, "patches.f32"), x)
        writeF32(File(work, "sigmas.f32"), animaSigmas(steps))
        val step = Regex("^draw step (\\d+):")
        run(ctx, listOf(File(wdir, "anima_q8.gguf").absolutePath, work.absolutePath, "0", "${width}x$height"),
            mapOf("DRAW" to "$steps", "CEN" to File(wdir, "anima_q8.cen").absolutePath,
                "TUNEFILE" to File(home, "anima.tune").absolutePath)) { l ->
            step.find(l)?.let { progress.onStep(it.groupValues[1].toInt() + 1, steps) }
        }
        // patches -> (2hp, 2wp, 16), channels innermost, for the small decoder
        val z = readF32(File(work, "draw.f32"))
        if (z.size != hp * wp * 68) error("graphics-chip engine gave ${z.size} numbers, not ${hp * wp * 68}")
        val l2 = FloatArray(4 * hp * wp * 16)
        for (y in 0 until 2 * hp) for (xx in 0 until 2 * wp) for (c in 0 until 16)
            l2[(y * 2 * wp + xx) * 16 + c] = z[((y / 2) * wp + xx / 2) * 68 + c * 4 + (y % 2) * 2 + xx % 2]
        val dec = File(work, "dec").apply { mkdirs() }
        writeF32(File(dec, "lat.f32"), l2)
        File(dec, "info.txt").writeText("${2 * hp} ${2 * wp}\n")
        run(ctx, listOf(File(wdir, "tae_w21.gguf").absolutePath, dec.absolutePath),
            mapOf("TUNEFILE" to File(home, "tae_w21.tune").absolutePath))
        val rgb = readF32(File(dec, "gpu_rgb.f32"))
        if (rgb.size != width * height * 3) error("picture decoder gave ${rgb.size} numbers, not ${width * height * 3}")
        work.deleteRecursively()
        fun px(v: Float) = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        return IntArray(width * height) { k -> (0xFF shl 24) or (px(rgb[3 * k]) shl 16) or (px(rgb[3 * k + 1]) shl 8) or px(rgb[3 * k + 2]) }
    }

    /** ComfyUI's ModelSamplingDiscreteFlow (shift 3) table + the "simple" scheduler, as the AI-chip route. */
    private fun animaSigmas(steps: Int): FloatArray {
        val tab = FloatArray(1000) { i -> val t = (i + 1) / 1000.0; (3.0 * t / (1.0 + 2.0 * t)).toFloat() }
        val ss = 1000.0 / steps
        return FloatArray(steps + 1) { k -> if (k == steps) 0f else tab[999 - (k * ss).toInt()] }
    }

    // ---- FLUX.2 klein 4B ----
    private const val KLEIN_FILES = "klein_q8.gguf tae_f2.gguf taee_f2.gguf klein_reader.gguf qwen_tokenizer.json"

    fun kleinDir(ctx: Context): File = ModelStore.KLEIN.gpu!!.modelDir(ctx)

    fun kleinReady(ctx: Context) = program(ctx).isFile && File(ctx.applicationInfo.nativeLibraryDir, "libtexttaps.so").isFile &&
        KLEIN_FILES.split(" ").all { File(kleinDir(ctx), it).isFile }

    /** Klein on the graphics chip: picked on the models page, or the phone has no AI-chip build of it. */
    fun useForKlein(ctx: Context) = ModelStore.KLEIN.runsOnGpu(ctx) && kleinReady(ctx)

    /** One rotation row: axis 0 reference index, 1 row, 2 column, 3 prompt index; 16 frequencies each, theta 2000.
     *  Engine order: P = [cos | cos], Q = [-sin | sin] over the 64 pairs. */
    private fun kleinRope(P: FloatArray, Q: FloatArray, row: Int, id: DoubleArray) {
        for (a in 0 until 4) for (i in 0 until 16) {
            val ang = id[a] * Math.pow(2000.0, -(2.0 * i) / 32.0)
            val k = row * 128 + a * 16 + i
            val c = Math.cos(ang).toFloat(); val sn = Math.sin(ang).toFloat()
            P[k] = c; P[k + 64] = c; Q[k] = -sn; Q[k + 64] = sn
        }
    }

    /**
     * [width] x [height] ARGB pixels from the chat-wrapped prompt [ids] (KleinTokens.encode). The reader (Qwen3-4B,
     * 8-bit, on the processor: libtexttaps.so) gives three of its inner layers; with [edit] (same size) the small
     * Flux 2 encoder turns that picture into extra rows the drawing looks at. Progress: 0 while reading, then the steps.
     */
    fun kleinGenerate(ctx: Context, ids: IntArray, seed: Long, steps: Int, width: Int, height: Int, edit: IntArray?,
                      progress: Progress): IntArray {
        val h = height / 16; val w = width / 16; val Li = h * w; val C = 128
        if (edit != null && edit.size != width * height) error("the picture to edit has the wrong size")
        val wdir = kleinDir(ctx)
        progress.onStep(0, steps)
        val work = File(ctx.cacheDir, "ugpu_klein").apply { deleteRecursively(); mkdirs() }
        val home = home(ctx)
        // 1. the reader
        val ib = ByteBuffer.allocate(ids.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        ib.asIntBuffer().put(ids)
        File(work, "ids.i32").writeBytes(ib.array())
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8).toString()
        runProgram(ctx, "libtexttaps.so", listOf(File(wdir, "klein_reader.gguf").absolutePath,
            File(work, "ids.i32").absolutePath, work.absolutePath, threads, "klein"), emptyMap())
        // 2. the picture to edit: small encoder (picture in [0, 1], a zero 4th channel), packed 2 x 2 like the drawing
        val Lr = if (edit != null) Li else 0
        if (edit != null) {
            val enc = File(work, "enc").apply { mkdirs() }
            val pic = FloatArray(width * height * 4)
            for (i in edit.indices) { val c = edit[i]
                pic[4 * i] = ((c shr 16) and 255) / 255f; pic[4 * i + 1] = ((c shr 8) and 255) / 255f; pic[4 * i + 2] = (c and 255) / 255f }
            writeF32(File(enc, "lat.f32"), pic)
            File(enc, "info.txt").writeText("$height $width\n")
            run(ctx, listOf(File(wdir, "taee_f2.gguf").absolutePath, enc.absolutePath),
                mapOf("TUNEFILE" to File(home, "taee_f2.tune").absolutePath))
            val e = readF32(File(enc, "gpu_rgb.f32"))   // (2h, 2w, 32)
            if (e.size != 4 * Li * 32) error("picture encoder gave ${e.size} numbers, not ${4 * Li * 32}")
            val ref = FloatArray(Li * C)
            for (y in 0 until h) for (x in 0 until w) for (c in 0 until 32) for (i in 0 until 2) for (j in 0 until 2)
                ref[((y * w + x) * 32 + c) * 4 + i * 2 + j] = e[((2 * y + i) * (2 * w) + 2 * x + j) * 32 + c]
            writeF32(File(work, "ref.f32"), ref)
        }
        // 3. positions, noise, schedule
        val Ni = Li + Lr; val Lt = ids.size
        val Lp = (Ni + Lt + 7) / 8 * 8 - Ni - Lt
        val N = Ni + Lt + Lp
        val P = FloatArray(N * 128) { 1f }; val Q = FloatArray(N * 128)
        for (r in 0 until (if (edit != null) 2 else 1)) for (y in 0 until h) for (x in 0 until w)
            kleinRope(P, Q, r * Li + y * w + x, doubleArrayOf(if (r == 1) 10.0 else 0.0, y.toDouble(), x.toDouble(), 0.0))
        for (t in 0 until Lt) kleinRope(P, Q, Ni + t, doubleArrayOf(0.0, 0.0, 0.0, t.toDouble()))
        writeF32(File(work, "rope_P.f32"), P); writeF32(File(work, "rope_Q.f32"), Q)
        writeF32(File(work, "sigmas.f32"), kleinSigmas(steps, Li))
        val rnd = java.util.Random(seed)
        writeF32(File(work, "noise.f32"), FloatArray(Li * C) { rnd.nextGaussian().toFloat() })
        writeF32(File(work, "img.f32"), FloatArray(Ni * C))
        writeF32(File(work, "sigma.f32"), floatArrayOf(1f))
        File(work, "info.txt").writeText("$Ni $Lt $Lp $width $height\n")
        // 4. the drawing, then the small decoder
        val step = Regex("^draw step (\\d+):")
        run(ctx, listOf(File(wdir, "klein_q8.gguf").absolutePath, work.absolutePath),
            mapOf("DRAW" to "1", "TUNEFILE" to File(home, "klein.tune").absolutePath)) { l ->
            step.find(l)?.let { progress.onStep(it.groupValues[1].toInt() + 1, steps) }
        }
        val out = decodeFlux2(ctx, File(work, "draw.f32"), h, w, wdir)
        work.deleteRecursively()
        return out
    }

    /** comfy_extras/nodes_flux.py get_schedule (empirical mu), as the AI-chip route. */
    private fun kleinSigmas(steps: Int, seq: Int): FloatArray {
        val a1 = 8.73809524e-05; val b1 = 1.89833333; val a2 = 0.00016927; val b2 = 0.45666666
        val mu = if (seq > 4300) a2 * seq + b2 else {
            val m200 = a2 * seq + b2; val m10 = a1 * seq + b1
            val a = (m200 - m10) / 190.0; a * steps + (m200 - 200.0 * a)
        }
        return FloatArray(steps + 1) { i ->
            val t = 1.0 - i.toDouble() / steps
            if (t <= 0) 0f else (Math.exp(mu) / (Math.exp(mu) + (1.0 / t - 1.0))).toFloat()
        }
    }

    // ---- Qwen-Image 2.1 Turbo ----
    private const val QWEN_FILES = "qwen_q4.gguf tae_qwen.gguf qimg_reader.gguf qwen_tokenizer.json"
    /** Turbo's 8 fixed noise levels (no shift), as the AI-chip route's sigmas.txt; the engine adds the final 0. */
    private val QWEN_SIGMAS = floatArrayOf(1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f)
    private const val QWEN_ROWS = 128   // prompt rows the drawing model holds

    fun qwenDir(ctx: Context): File = ModelStore.QWEN.gpu!!.modelDir(ctx)

    fun qwenReady(ctx: Context) = program(ctx).isFile && File(ctx.applicationInfo.nativeLibraryDir, "libtexttaps.so").isFile &&
        QWEN_FILES.split(" ").all { File(qwenDir(ctx), it).isFile }

    /** Qwen-Image on the graphics chip: picked on the models page, or the phone has no AI-chip build of it. */
    fun useForQwen(ctx: Context) = ModelStore.QWEN.runsOnGpu(ctx) && qwenReady(ctx)

    /**
     * [width] x [height] ARGB pixels from the chat-wrapped prompt [ids] (QwenImageTokens.encode, no picture). The
     * reader (Qwen3-VL-8B's language half, 4-bit, on the processor: libtexttaps.so) gives its last layer before the
     * final norm; the drawing sees the rows from the user turn on (the system turn dropped, as the AI-chip route).
     */
    fun qwenGenerate(ctx: Context, ids: IntArray, seed: Long, width: Int, height: Int, progress: Progress): IntArray {
        val h = height / 16; val w = width / 16; val Ni = h * w; val C = 64; val D = 4096
        val steps = QWEN_SIGMAS.size
        val wdir = qwenDir(ctx)
        progress.onStep(0, steps)
        var s0 = -1; var seen = 0   // the user turn starts at the second <|im_start|>
        for (i in ids.indices) if (ids[i] == 151644 && ++seen == 2) { s0 = i; break }
        if (s0 < 0) error("prompt is not in the chat template")
        val nc = ids.size - s0
        if (nc > QWEN_ROWS) error("prompt longer than ${QWEN_ROWS - 6} words")
        val work = File(ctx.cacheDir, "ugpu_qwen").apply { deleteRecursively(); mkdirs() }
        val home = home(ctx)
        // 1. the reader
        val ib = ByteBuffer.allocate(ids.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        ib.asIntBuffer().put(ids)
        File(work, "ids.i32").writeBytes(ib.array())
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8).toString()
        runProgram(ctx, "libtexttaps.so", listOf(File(wdir, "qimg_reader.gguf").absolutePath,
            File(work, "ids.i32").absolutePath, work.absolutePath, threads, "qimg"), emptyMap())
        val hid = readF32(File(work, "hidden.f32"))
        if (hid.size != ids.size * D) error("prompt reader gave ${hid.size} numbers, not ${ids.size * D}")
        val c = FloatArray(QWEN_ROWS * D)
        System.arraycopy(hid, s0 * D, c, 0, nc * D)
        writeF32(File(work, "ctx.f32"), c)
        File(work, "hidden.f32").delete()
        // 2. noise and schedule
        val rnd = java.util.Random(seed)
        writeF32(File(work, "noise.f32"), FloatArray(Ni * C) { rnd.nextGaussian().toFloat() })
        writeF32(File(work, "sigmas.f32"), QWEN_SIGMAS)
        File(work, "info.txt").writeText("$nc $width $height\n")
        // 3. the drawing (blocks streamed from storage: the whole model would not fit next to the app), then the decoder
        val step = Regex("^draw step (\\d+):")
        run(ctx, listOf(File(wdir, "qwen_q4.gguf").absolutePath, work.absolutePath),
            mapOf("DRAW" to "1", "STREAM" to "1", "TUNEFILE" to File(home, "qwen.tune").absolutePath)) { l ->
            step.find(l)?.let { progress.onStep(it.groupValues[1].toInt() + 1, steps) }
        }
        val z = readF32(File(work, "draw.f32"))
        if (z.size != Ni * C) error("graphics-chip engine gave ${z.size} numbers, not ${Ni * C}")
        val dec = File(work, "dec").apply { mkdirs() }
        File(work, "draw.f32").renameTo(File(dec, "lat.f32"))   // rows (y, x, 64) are the decoder's input as they are
        File(dec, "info.txt").writeText("$h $w\n")
        run(ctx, listOf(File(wdir, "tae_qwen.gguf").absolutePath, dec.absolutePath),
            mapOf("TUNEFILE" to File(home, "tae_qwen.tune").absolutePath))
        val rgb = readF32(File(dec, "gpu_rgb.f32"))
        if (rgb.size != width * height * 3) error("picture decoder gave ${rgb.size} numbers, not ${width * height * 3}")
        work.deleteRecursively()
        fun px(v: Float) = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        return IntArray(width * height) { k ->
            (0xFF shl 24) or (px(rgb[3 * k]) shl 16) or (px(rgb[3 * k + 1]) shl 8) or px(rgb[3 * k + 2])
        }
    }

    // ---- Wan 2.2 TI2V-5B Turbo: picture to clip ----
    private const val WAN_FILES = "wan_q8.gguf wanenc.gguf tae_w22.gguf umt5_reader.gguf"
    /** Turbo's 4 levels; each step predicts the clean clip and re-noises it (the engine's default "lcm" sampler). */
    private val WAN_SIGMAS = floatArrayOf(1.0f, 0.9375f, 0.833333f, 0.625f, 0f)
    private const val WAN_PROMPT_ROWS = 512   // ComfyUI pads the prompt to 512 rows with zeros
    // the 5B model's latents are rescaled per channel (ComfyUI latent_formats.Wan22): model = (raw - mean) / std
    private val WAN_MEAN = floatArrayOf(-0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f, -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.1570f, -0.0098f, 0.0375f, -0.1825f, -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f, -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.1230f, -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.0520f, 0.3748f, 0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f)
    private val WAN_STD = floatArrayOf(0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f, 0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f, 0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f, 0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f, 0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f, 0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f)

    fun wanDir(ctx: Context): File = ModelStore.WAN.gpu!!.modelDir(ctx)

    fun wanReady(ctx: Context) = program(ctx).isFile && WAN_FILES.split(" ").all { File(wanDir(ctx), it).isFile }

    /** Wan on the graphics chip: picked on the models page, or the phone has no AI-chip build of it. */
    fun useForWan(ctx: Context) = ModelStore.WAN.runsOnGpu(ctx) && wanReady(ctx)

    /** Frames per clip: frames.txt in the graphics folder (4n + 1), else 17 (about 1 s at 16 per second). */
    fun wanFrames(ctx: Context): Int =
        File(wanDir(ctx), "frames.txt").takeIf { it.isFile }?.readText()?.trim()?.toIntOrNull()?.takeIf { it >= 5 && (it - 1) % 4 == 0 } ?: 17

    /**
     * A clip from the start picture [start] (768 x 768 ARGB) and the prompt's reader rows [rows] (n x 4096, WanReader).
     * The full Wan VAE encoder (wanenc) makes the picture's latent frame, the engine draws the clip with that frame
     * held, and the small decoder (lighttaew2_2) gives every frame. Returns the frames, 768 x 768 ARGB each.
     */
    fun wanGenerate(ctx: Context, rows: FloatArray, start: IntArray, seed: Long, progress: Progress): List<IntArray> {
        val W = 768; val H = 768; val D = 4096
        if (start.size != W * H) error("the start picture must be 768 x 768")
        val FR = wanFrames(ctx); val T = (FR - 1) / 4 + 1
        val hl = H / 16; val wl = W / 16; val Hp = hl / 2; val Wp = wl / 2; val L = T * Hp * Wp
        val steps = WAN_SIGMAS.size - 1
        val wdir = wanDir(ctx)
        progress.onStep(0, steps)
        val work = File(ctx.cacheDir, "ugpu_wan").apply { deleteRecursively(); mkdirs() }
        val home = home(ctx)
        // 1. the start picture -> raw latent mu (hl, wl, 48): input = picture * 2 - 1, unshuffled 2 x 2 (c * 4 + dy * 2 + dx)
        val enc = File(work, "enc").apply { mkdirs() }
        val h2 = H / 2; val w2 = W / 2
        val pin = FloatArray(h2 * w2 * 12)
        for (y in 0 until H) for (x in 0 until W) {
            val c = start[y * W + x]
            val v = floatArrayOf(((c shr 16) and 255) / 255f, ((c shr 8) and 255) / 255f, (c and 255) / 255f)
            for (ch in 0 until 3) pin[((y / 2) * w2 + x / 2) * 12 + ch * 4 + (y % 2) * 2 + (x % 2)] = v[ch] * 2f - 1f
        }
        writeF32(File(enc, "lat.f32"), pin)
        File(enc, "info.txt").writeText("$h2 $w2\n")
        run(ctx, listOf(File(wdir, "wanenc.gguf").absolutePath, enc.absolutePath), mapOf("TUNEFILE" to File(home, "wanenc.tune").absolutePath))
        val mu = readF32(File(enc, "gpu_rgb.f32"))
        if (mu.size != hl * wl * 48) error("start-picture encoder gave ${mu.size} numbers, not ${hl * wl * 48}")
        // 2. its frame as the model's patches (hp, wp) x [c][dy][dx], rescaled
        val first = FloatArray(Hp * Wp * 192)
        for (hp in 0 until Hp) for (wp in 0 until Wp) for (c in 0 until 48) for (dy in 0 until 2) for (dx in 0 until 2)
            first[(hp * Wp + wp) * 192 + c * 4 + dy * 2 + dx] = (mu[((2 * hp + dy) * wl + 2 * wp + dx) * 48 + c] - WAN_MEAN[c]) / WAN_STD[c]
        writeF32(File(work, "first.f32"), first)
        // 3. prompt rows (zero rows up to 512), positions (frame, row, column: 44 / 42 / 42 channels, theta 10000),
        //    noise, levels
        val nr = minOf(rows.size / D, WAN_PROMPT_ROWS)
        val c = FloatArray(WAN_PROMPT_ROWS * D)
        System.arraycopy(rows, 0, c, 0, nr * D)
        writeF32(File(work, "ctx.f32"), c)
        val P = FloatArray(L * 128); val Q = FloatArray(L * 128)
        val dims = intArrayOf(44, 42, 42)
        var n = 0
        for (t in 0 until T) for (y in 0 until Hp) for (x in 0 until Wp) {
            var k = 0
            for ((a, pos) in intArrayOf(t, y, x).withIndex()) for (i in 0 until dims[a] / 2) {
                val ang = pos * Math.pow(10000.0, -2.0 * i / dims[a])
                val co = Math.cos(ang).toFloat(); val si = Math.sin(ang).toFloat()
                P[n * 128 + k] = co; P[n * 128 + 64 + k] = co; Q[n * 128 + k] = -si; Q[n * 128 + 64 + k] = si
                k++
            }
            n++
        }
        writeF32(File(work, "rope_P.f32"), P); writeF32(File(work, "rope_Q.f32"), Q)
        val rnd = java.util.Random(seed)
        val noise = FloatArray(L * 192) { rnd.nextGaussian().toFloat() }
        writeF32(File(work, "noise.f32"), noise)
        writeF32(File(work, "img.f32"), noise)   // read by the engine's one-step check only
        writeF32(File(work, "sigmas.f32"), WAN_SIGMAS)
        writeF32(File(work, "sigma.f32"), floatArrayOf(1f))
        File(work, "info.txt").writeText("$L $WAN_PROMPT_ROWS $T $Hp $Wp $W $H $FR\n")
        // 4. the drawing (blocks streamed from storage), then every frame through the small decoder
        val step = Regex("^draw step (\\d+):")
        run(ctx, listOf(File(wdir, "wan_q8.gguf").absolutePath, work.absolutePath),
            mapOf("DRAW" to "1", "STREAM" to "1", "SEED" to (seed and 0x7fffffff).toString(),
                "TUNEFILE" to File(home, "wan.tune").absolutePath)) { l ->
            step.find(l)?.let { progress.onStep(it.groupValues[1].toInt() + 1, steps) }
        }
        val z = readF32(File(work, "draw.f32"))
        if (z.size != L * 192) error("graphics-chip engine gave ${z.size} numbers, not ${L * 192}")
        val dec = File(work, "dec").apply { mkdirs() }
        val lat = FloatArray(T * hl * wl * 48)
        for (t in 0 until T) for (hp in 0 until Hp) for (wp in 0 until Wp) for (ch in 0 until 48) for (dy in 0 until 2) for (dx in 0 until 2)
            lat[((t * hl + 2 * hp + dy) * wl + 2 * wp + dx) * 48 + ch] =
                z[((t * Hp + hp) * Wp + wp) * 192 + ch * 4 + dy * 2 + dx] * WAN_STD[ch] + WAN_MEAN[ch]
        writeF32(File(dec, "lat.f32"), lat)
        File(dec, "info.txt").writeText("$hl $wl $T\n")
        run(ctx, listOf(File(wdir, "tae_w22.gguf").absolutePath, dec.absolutePath), mapOf("TUNEFILE" to File(home, "tae_w22.tune").absolutePath))
        val rgb = readF32(File(dec, "gpu_rgb.f32"))
        val NP = W * H
        if (rgb.size != FR * NP * 3) error("clip decoder gave ${rgb.size} numbers, not ${FR * NP * 3}")
        work.deleteRecursively()
        fun px(v: Float) = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        return List(FR) { f -> IntArray(NP) { k -> val o = (f * NP + k) * 3
            (0xFF shl 24) or (px(rgb[o]) shl 16) or (px(rgb[o + 1]) shl 8) or px(rgb[o + 2]) } }
    }
}
