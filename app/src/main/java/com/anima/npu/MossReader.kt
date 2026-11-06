package com.anima.npu

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * MOSS-Image's reader: Qwen 3.5 9B (models/moss/moss_reader.gguf) run by the app's own llama.cpp server on the AI chip
 * in embedding mode, which returns each word's last hidden state: what ComfyUI's MOSS reader gives. 4-bit (Q4_K_M with an
 * importance matrix), with the parts MOSS never uses taken out (the word-picking table and the faster-writing block):
 * 4.8 GB, so it fits a 12 GB phone. Its rows are 12% from ComfyUI's, but the pictures match the 8-bit reader's
 * (2026-11-06: 1.6% mean pixel difference on the phone, same scene). Started for one prompt and stopped again,
 * so its memory is free before drawing. The empty prompt's rows never change and are kept in empty_rows.f32.
 */
object MossReader {
    const val PORT = 8283
    const val FILE = "moss_reader.gguf"
    private const val W = 4096

    private fun text(prompt: String) = "<|im_start|>user\n${prompt.trim()}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    /** (prompt rows, empty-prompt rows), each rows x 4096. [say] gets progress lines for the screen. */
    fun read(ctx: Context, dir: File, prompt: String, say: (String) -> Unit = {}): Pair<FloatArray, FloatArray> {
        val cache = File(dir, "empty_rows.f32")
        var empty = runCatching { load(cache) }.getOrNull()
        val p = start(ctx, File(dir, FILE), say)
        try {
            say("Reading the prompt…")
            val cond = embed(text(prompt))
            // the 8-step build (sampler.txt "euler") draws without the empty prompt
            if (File(dir, "sampler.txt").takeIf { it.isFile }?.readText()?.trim() == "euler") return cond to FloatArray(0)
            if (empty == null) { empty = embed(text("")); save(cache, empty) }
            return cond to empty
        } finally {
            p.destroy(); runCatching { p.waitFor() }
        }
    }

    private fun start(ctx: Context, model: File, say: (String) -> Unit): Process {
        if (!model.isFile) error("MOSS reader missing: ${model.path}")
        LocalLing.stop()   // the chat model's memory first: this reader is 4.8 GB
        val libs = ctx.applicationInfo.nativeLibraryDir
        val cmd = listOf(
            "$libs/libllama-server.so", "-m", model.absolutePath, "--embeddings", "--pooling", "none",
            // the AI chip (without it llama.cpp picks the graphics chip, which fails): iQOO 13, loads in ~12 s,
            // reads a prompt in 0.4 s; phone memory peaks at 7.5 GB while drawing (2026-11-06)
            "--device", "HTP0", "-ngl", "99",
            "-c", "1024", "-np", "1", "-b", "512", "-ub", "512",
            "--host", "127.0.0.1", "--port", PORT.toString(), "--no-webui",
        )
        val pb = ProcessBuilder(cmd)
        pb.environment()["LD_LIBRARY_PATH"] = "$libs:/system/lib64:/vendor/lib64:/system/vendor/lib64"
        pb.environment()["ADSP_LIBRARY_PATH"] = libs
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val last = ArrayDeque<String>()
        Thread {
            runCatching {
                proc.inputStream.bufferedReader().forEachLine { l ->
                    if (l.isNotBlank()) { Log.i("mossreader", l.take(300)); synchronized(last) { last.addLast(l.take(200)); if (last.size > 5) last.removeFirst() } }
                }
            }
        }.apply { isDaemon = true }.start()
        say("Loading the MOSS reader…")
        val until = System.currentTimeMillis() + 180_000
        while (System.currentTimeMillis() < until) {
            if (!proc.isAlive) error("MOSS reader stopped while loading: ${synchronized(last) { last.lastOrNull() }}")
            if (healthy()) return proc
            Thread.sleep(300)
        }
        proc.destroy()
        error("MOSS reader did not start within 180 s: ${synchronized(last) { last.lastOrNull() }}")
    }

    private fun healthy() = try {
        val c = URL("http://127.0.0.1:$PORT/health").openConnection() as HttpURLConnection
        c.connectTimeout = 500; c.readTimeout = 1000
        val ok = c.responseCode in 200..299
        c.disconnect(); ok
    } catch (e: Exception) { false }

    /** The word rows of [s] (the server splits it into words itself, special words included: same ids as ComfyUI). */
    private fun embed(s: String): FloatArray {
        val c = URL("http://127.0.0.1:$PORT/embeddings").openConnection() as HttpURLConnection
        c.connectTimeout = 2000; c.readTimeout = 120_000
        c.requestMethod = "POST"; c.doOutput = true
        c.setRequestProperty("Content-Type", "application/json")
        c.outputStream.use { it.write(JSONObject().put("content", s).toString().toByteArray()) }
        if (c.responseCode !in 200..299) error("MOSS reader answered ${c.responseCode}: ${c.errorStream?.bufferedReader()?.readText()?.take(200)}")
        val body = c.inputStream.bufferedReader().readText()
        c.disconnect()
        val rows = JSONArray(body).getJSONObject(0).getJSONArray("embedding")
        val out = FloatArray(rows.length() * W)
        for (i in 0 until rows.length()) {
            val r = rows.getJSONArray(i)
            if (r.length() != W) error("MOSS reader gave rows of ${r.length()}, not $W")
            for (k in 0 until W) out[i * W + k] = r.getDouble(k).toFloat()
        }
        return out
    }

    private fun save(f: File, a: FloatArray) = DataOutputStream(f.outputStream().buffered()).use { o ->
        o.writeInt(a.size); for (v in a) o.writeFloat(v)
    }

    private fun load(f: File): FloatArray = DataInputStream(f.inputStream().buffered()).use { i ->
        FloatArray(i.readInt()) { i.readFloat() }
    }
}
