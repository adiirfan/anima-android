package com.anima.npu

import android.graphics.Bitmap
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Checks the phone against the computer. The computer writes, per test prompt, the tokenizer ids and
 * the exact starting noise into files/selftest/ (push_models.sh); the app draws each one and writes the
 * final latent, the picture and the stage timings into files/selftest/out/, ending with a DONE line.
 */
object SelfTest {
    private fun ints(f: File): IntArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
        return IntArray(b.remaining()).also { b.get(it) }
    }

    private fun floats(f: File): FloatArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(b.remaining()).also { b.get(it) }
    }

    /** One prompt per line in, "qwen ids | t5 ids | t5 weights" per line out, for comparing with ComfyUI. */
    private fun tokenCases(root: File, inFile: File, outFile: File) {
        if (!inFile.isFile) return
        Tokens.init(File(root, "models"))
        outFile.writeText(inFile.readLines().filter { it.isNotEmpty() }.joinToString("\n") { p ->
            val r = Tokens.encode(p)
            r.qwen.joinToString(" ") + " | " + r.t5.joinToString(" ") + " | " + r.t5Weights.joinToString(" ") { "%.4f".format(it) }
        } + "\n")
    }

    fun run(root: File, log: (String) -> Unit) {
        val dir = File(root, "selftest")
        val out = File(dir, "out").apply { deleteRecursively(); mkdirs() }
        val timing = StringBuilder()
        val cases = File(dir, "cases.txt")
        for (line in (if (cases.isFile) cases.readLines() else emptyList()).filter { it.isNotBlank() }) {
            val (seed, steps) = line.trim().split(" ").map { it.toLong() }
            val t5 = ints(File(dir, "t5_$seed.i32"))
            val px = Engine.generate(
                ints(File(dir, "qwen_$seed.i32")), t5, FloatArray(t5.size) { 1f }, seed, steps.toInt(), 512, 512,
                null, 1f, 0.5f, floats(File(dir, "noise_$seed.f32")),
            ) { s, n -> log("seed $seed: step $s of $n") }
            if (px == null) {
                timing.append("seed $seed ERROR ${Engine.lastError()}\n")
                log("seed $seed: ${Engine.lastError()}")
                continue
            }
            val t = Engine.lastTimes()
            val lat = Engine.lastLatent()
            val bb = ByteBuffer.allocate(lat.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            bb.asFloatBuffer().put(lat)
            File(out, "latent_$seed.f32").writeBytes(bb.array())
            val bmp = Bitmap.createBitmap(px, 512, 512, Bitmap.Config.ARGB_8888)
            File(out, "rgb_$seed.png").outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
            val row = "seed $seed steps $steps prompt %.0f ms, loading %.0f ms, drawing %.0f ms, decode %.0f ms"
                .format(t[1], t[0], t[2], t[3])
            timing.append(row).append('\n')
            log(row)
        }
        tokenCases(root, File(dir, "token_cases.txt"), File(out, "token_ids.txt"))
        timing.append("DONE\n")
        File(out, "timing.txt").writeText(timing.toString())
    }
}
