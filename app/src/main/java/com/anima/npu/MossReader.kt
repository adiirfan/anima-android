package com.anima.npu

import android.content.Context
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

/**
 * MOSS-Image's reader: Qwen 3.5 9B (models/moss/moss_reader.gguf), its last layer's rows after the final norm (what
 * ComfyUI's MOSS reader gives), on the AI chip. 4-bit (Q4_K_M with an importance matrix), with the parts MOSS never uses
 * taken out (the word-picking table and the faster-writing block): 4.8 GB, so it fits a 12 GB phone. Its rows are 12%
 * from ComfyUI's, but the pictures match the 8-bit reader's (2026-11-06: 1.6% mean pixel difference, same scene).
 * Run by the app's reader program (libtexttaps.so, mode "last", the prompt as text) for one prompt, so its memory is
 * free before drawing. It used to be llama.cpp's server, asked over a connection to the phone itself, and some phones
 * refused that connection without internet (GitHub issue 1, OnePlus 15); the program writes a file instead (its rows
 * are identical to the server's). The empty prompt's rows never change and are kept in empty_rows.f32.
 */
object MossReader {
    const val FILE = "moss_reader.gguf"
    private const val W = 4096

    private fun text(prompt: String) = "<|im_start|>user\n${prompt.trim()}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    /** (prompt rows, empty-prompt rows), each rows x 4096. [say] gets progress lines for the screen. */
    fun read(ctx: Context, dir: File, prompt: String, say: (String) -> Unit = {}, aiChip: Boolean = true): Pair<FloatArray, FloatArray> {
        val model = File(dir, FILE)
        if (!model.isFile) error("MOSS reader missing: ${model.path}")
        LocalLing.stop()   // the chat model's memory first: this reader is 4.8 GB
        val cache = File(dir, "empty_rows.f32")
        var empty = runCatching { load(cache) }.getOrNull()
        val cond = embed(ctx, model, text(prompt), say, aiChip)
        // the 8-step build (sampler.txt "euler") draws without the empty prompt
        if (File(dir, "sampler.txt").takeIf { it.isFile }?.readText()?.trim() == "euler") return cond to FloatArray(0)
        if (empty == null) { empty = embed(ctx, model, text(""), say, aiChip); save(cache, empty) }
        return cond to empty
    }

    /** The reader's rows for [s] (split into words by the program, special words included: same ids as ComfyUI). */
    private fun embed(ctx: Context, model: File, s: String, say: (String) -> Unit, aiChip: Boolean): FloatArray {
        val work = File(ctx.cacheDir, "moss_reader").apply { deleteRecursively(); mkdirs() }
        try {
            File(work, "prompt.txt").writeText(s)
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8).toString()
            // the AI chip (iQOO 13: loads in ~12 s, reads a prompt in 0.4 s), or the processor on phones without it
            val env = if (aiChip) mapOf("READER_HTP" to "1", "ADSP_LIBRARY_PATH" to ctx.applicationInfo.nativeLibraryDir) else emptyMap()
            say("Loading the MOSS reader…")
            GpuEngine.runProgram(ctx, "libtexttaps.so", listOf(model.absolutePath, File(work, "prompt.txt").absolutePath,
                work.absolutePath, threads, "last"), env) { l -> if (l.startsWith("loaded in")) say("Reading the prompt…") }
            val rows = GpuEngine.readF32(File(work, "hidden.f32"))
            if (rows.isEmpty() || rows.size % W != 0) error("MOSS reader gave ${rows.size} numbers, not rows of $W")
            return rows
        } finally {
            work.deleteRecursively()
        }
    }

    private fun save(f: File, a: FloatArray) = DataOutputStream(f.outputStream().buffered()).use { o ->
        o.writeInt(a.size); for (v in a) o.writeFloat(v)
    }

    private fun load(f: File): FloatArray = DataInputStream(f.inputStream().buffered()).use { i ->
        FloatArray(i.readInt()) { i.readFloat() }
    }
}
