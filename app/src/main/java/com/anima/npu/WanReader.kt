package com.anima.npu

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Wan 2.2's prompt reader: umT5-XXL (models/wan/umt5_reader.gguf, 4-bit, 3.7 GB) run by the app's own llama.cpp server
 * in embedding mode, which returns each word's last state: what ComfyUI's Wan reader gives (2026-11-07: 0.985 average
 * likeness to ComfyUI's rows on the same prompt). The file is marked to read the prompt in both directions, as the
 * model was trained; the server has no option for that. Started for one prompt and stopped again, so its memory is free
 * before the clip is made. Wan never uses an empty prompt (no guidance in the 4-step version).
 */
object WanReader {
    const val PORT = 8284
    const val FILE = "umt5_reader.gguf"
    private const val W = 4096

    /** The prompt's rows, n x 4096 (the reader adds its own end word). [say] gets progress lines for the screen. */
    fun read(ctx: Context, dir: File, prompt: String, say: (String) -> Unit = {}): FloatArray {
        val p = start(ctx, File(dir, FILE), say)
        try {
            say("Reading the prompt…")
            return embed(prompt.trim())
        } finally {
            p.destroy(); runCatching { p.waitFor() }
        }
    }

    private fun start(ctx: Context, model: File, say: (String) -> Unit): Process {
        if (!model.isFile) error("Wan prompt reader missing: ${model.path}")
        LocalLing.stop()   // the chat model's memory first
        val libs = ctx.applicationInfo.nativeLibraryDir
        val cmd = listOf(
            "$libs/libllama-server.so", "-m", model.absolutePath, "--embeddings", "--pooling", "none",
            // the processor, not the AI chip: on the chip this reader comes back blank (the app's llama.cpp) or
            // unrelated to ComfyUI's (0.26 likeness, an older build); the processor matches the computer at 0.999
            // and reads a prompt in ~1 s after ~10 s loading (iQOO 13, 2026-11-07)
            "--device", "none", "-ngl", "0", "-t", "6",
            "-c", "512", "-np", "1", "-b", "512", "-ub", "512",
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
                    if (l.isNotBlank()) { Log.i("wanreader", l.take(300)); synchronized(last) { last.addLast(l.take(200)); if (last.size > 5) last.removeFirst() } }
                }
            }
        }.apply { isDaemon = true }.start()
        say("Loading the Wan prompt reader…")
        val until = System.currentTimeMillis() + 180_000
        while (System.currentTimeMillis() < until) {
            if (!proc.isAlive) error("Wan prompt reader stopped while loading: ${synchronized(last) { last.lastOrNull() }}")
            if (healthy()) return proc
            Thread.sleep(300)
        }
        proc.destroy()
        error("Wan prompt reader did not start within 180 s: ${synchronized(last) { last.lastOrNull() }}")
    }

    private fun healthy() = try {
        val c = URL("http://127.0.0.1:$PORT/health").openConnection() as HttpURLConnection
        c.connectTimeout = 500; c.readTimeout = 1000
        val ok = c.responseCode in 200..299
        c.disconnect(); ok
    } catch (e: Exception) { false }

    private fun embed(s: String): FloatArray {
        val c = URL("http://127.0.0.1:$PORT/embeddings").openConnection() as HttpURLConnection
        c.connectTimeout = 2000; c.readTimeout = 120_000
        c.requestMethod = "POST"; c.doOutput = true
        c.setRequestProperty("Content-Type", "application/json")
        c.outputStream.use { it.write(JSONObject().put("content", s).toString().toByteArray()) }
        if (c.responseCode !in 200..299) error("Wan prompt reader answered ${c.responseCode}: ${c.errorStream?.bufferedReader()?.readText()?.take(200)}")
        val body = c.inputStream.bufferedReader().readText()
        c.disconnect()
        val rows = JSONArray(body).getJSONObject(0).getJSONArray("embedding")
        val n = minOf(rows.length(), 512)
        val out = FloatArray(n * W)
        for (i in 0 until n) {
            val r = rows.getJSONArray(i)
            if (r.length() != W) error("Wan prompt reader gave rows of ${r.length()}, not $W")
            for (k in 0 until W) out[i * W + k] = r.getDouble(k).toFloat()
        }
        return out
    }
}
