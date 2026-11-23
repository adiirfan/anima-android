package com.anima.npu

import android.content.Context
import java.io.File

/**
 * Wan 2.2's prompt reader: umT5-XXL (models/wan/umt5_reader.gguf, 4-bit, 3.7 GB), each word's last state: what
 * ComfyUI's Wan reader gives (2026-11-07: 0.985 average likeness to ComfyUI's rows on the same prompt). The file is
 * marked to read the prompt in both directions, as the model was trained. Run by the app's reader program
 * (libtexttaps.so, mode "last", the prompt as text) for one prompt, so its memory is free before the clip is made. It
 * used to be llama.cpp's server asked over a connection to the phone itself, which some phones refuse without internet
 * (GitHub issue 1); the program's rows are identical to the server's (2026-11-23). On the processor, not the AI chip: on
 * the chip this reader came back blank or unrelated to ComfyUI's; the processor matches the computer at 0.999 and reads
 * a prompt in ~1 s after ~10 s loading (iQOO 13). Wan never uses an empty prompt (no guidance in the 4-step version).
 */
object WanReader {
    const val FILE = "umt5_reader.gguf"
    private const val W = 4096

    /** The prompt's rows, n x 4096 (the reader adds its own end word). [say] gets progress lines for the screen. */
    fun read(ctx: Context, dir: File, prompt: String, say: (String) -> Unit = {}): FloatArray {
        val model = File(dir, FILE)
        if (!model.isFile) error("Wan prompt reader missing: ${model.path}")
        LocalLing.stop()   // the chat model's memory first
        val work = File(ctx.cacheDir, "wan_reader").apply { deleteRecursively(); mkdirs() }
        try {
            File(work, "prompt.txt").writeText(prompt.trim())
            say("Loading the Wan prompt reader…")
            // no private rearranged copy of the weights (3.7 GB of memory): used straight from the file, which also
            // loads and reads a prompt faster here
            GpuEngine.runProgram(ctx, "libtexttaps.so", listOf(model.absolutePath, File(work, "prompt.txt").absolutePath,
                work.absolutePath, "6", "last"), mapOf("READER_NOCOPY" to "1")) { l ->
                if (l.startsWith("loaded in")) say("Reading the prompt…")
            }
            val rows = GpuEngine.readF32(File(work, "hidden.f32"))
            if (rows.isEmpty() || rows.size % W != 0) error("Wan prompt reader gave ${rows.size} numbers, not rows of $W")
            return rows
        } finally {
            work.deleteRecursively()
        }
    }
}
