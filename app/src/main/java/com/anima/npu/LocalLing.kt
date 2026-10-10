package com.anima.npu

import android.content.Context
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Ling-3.0-tiny on the phone's AI chip, answering like an online chat service: llama.cpp's own server, started as a
 * separate program and listening on the loopback address (the same arrangement as Gemma inside Mihon).
 *
 * A separate program rather than a library call: the chip work runs in its own process on the DSP either way, and a
 * model that crashes takes its own program down instead of the app. The chat then speaks plain HTTP to it.
 */
object LocalLing {
    /** Nothing else on the phone listens here (Mihon's Gemma uses 8181). */
    const val PORT = 8282
    const val BASE_URL = "http://127.0.0.1:$PORT"
    const val FILE = "Ling-3.0-tiny-Q4_0-npu2.gguf"
    const val SMALL_FILE = "Qwen3.5-0.8B-Q8_0.gguf"
    /** The "lingExperts" setting that picks the small chat model (Qwen3.5 0.8B) instead of Ling. */
    const val SMALL = -1

    /** The Lowest memory setting: Qwen3.5 0.8B instead of Ling, about 1.4 GB in all instead of 2.3 to 4.9. */
    fun small(ctx: Context) = expertSlots(ctx) == SMALL
    fun store(ctx: Context) = if (small(ctx)) ModelStore.SMALLCHAT else ModelStore.LING
    /** The chat model's name, for the screen. */
    fun name(ctx: Context) = if (small(ctx)) "Qwen3.5 0.8B" else "Ling"

    fun modelFile(ctx: Context) =
        if (small(ctx)) File(ModelStore.SMALLCHAT.modelDir(ctx), SMALL_FILE) else File(ModelStore.LING.modelDir(ctx), FILE)
    /** The experts repacked for the AI chip, read in as needed when streaming (made on the first start, 3.9 GB). */
    fun expertFile(ctx: Context) = File(ModelStore.LING.modelDir(ctx), "Ling-3.0-tiny-Q4_0-npu2.experts")

    /**
     * How many of each layer's 128 experts stay on the AI chip; 0 keeps them all there (fastest, about 4.5 GB
     * locked). Streaming the rest from the expert file costs speed but frees the memory for other apps: on the
     * S25 Edge 32 locked 1.8 GB instead of 4.6 and wrote 13 words a second instead of 25 (2026-10-09).
     */
    fun expertSlots(ctx: Context): Int = ctx.getSharedPreferences("anima", Context.MODE_PRIVATE).getInt("lingExperts", 32)

    fun setExpertSlots(ctx: Context, n: Int) {
        ctx.getSharedPreferences("anima", Context.MODE_PRIVATE).edit().putInt("lingExperts", n).apply()
        // the next message starts Ling with the new setting
        Thread { stop() }.start()
        if (n == 0) { expertFile(ctx).delete(); File(expertFile(ctx).path + ".done").delete() }
    }
    fun ready(ctx: Context): Boolean = modelFile(ctx).length() > 0

    @Volatile private var proc: Process? = null
    private val ring = ArrayDeque<String>()

    fun running(): Boolean = proc?.isAlive == true

    /** The server's last line of output: shown when it fails to start. */
    val lastLog: String get() = synchronized(ring) { ring.lastOrNull() ?: "" }
    fun log(): List<String> = synchronized(ring) { ring.toList() }

    private fun keep(line: String) = synchronized(ring) {
        ring.addLast(line.take(300))
        while (ring.size > 50) ring.removeFirst()
    }

    /**
     * Start it if it is not already up, and wait until it answers (blocking, up to 180 s; opening takes about 10 s).
     * [say] gets progress lines for the screen.
     */
    @Synchronized
    fun start(ctx: Context, say: (String) -> Unit = {}): Boolean {
        if (running() && answers()) return true
        stop()
        if (!ready(ctx)) { say("${name(ctx)} is not downloaded yet"); return false }
        val small = small(ctx)
        // Programs may only be run from the folder Android unpacks the libraries into, which is also where the chip
        // loads its own half of the work from.
        val libs = ctx.applicationInfo.nativeLibraryDir
        val cmd = listOf(
            "$libs/libllama-server.so", "-m", modelFile(ctx).absolutePath,
            // the AI chip; without this it quietly uses the processor
            "--device", "HTP0", "-ngl", "99",
            // one conversation at a time: the default 4 slots only cost memory
            "-c", "8192", "-np", "1",
            "--host", "127.0.0.1", "--port", PORT.toString(), "--alias", "ling",
            // the chat template carries the model's tool-call format
            "--jinja", "--no-webui",
        ) + if (small) listOf(
            // 512-word batches: 0.26 GB less than 2048 and faster to read on the chip (1962 vs 1534 words/s, iQOO 13)
            "-b", "512", "-ub", "512",
            // Qwen's suggested sampling without thinking
            "--temp", "0.7", "--top-p", "0.8", "--top-k", "20",
        ) else listOf(
            // 2048-word batches read a long prompt fastest on the chip (575 -> 633 words/s measured)
            "-b", "2048", "-ub", "2048",
            "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
        )
        val pb = ProcessBuilder(cmd)
        // llama.cpp asks for the graphics chip's library, which lives with the phone's firmware, whether or not it is
        // used: without these folders on the path the program does not start at all. The system folder goes before
        // the vendor's: the vendor's older libc++.so lacks symbols the Samsung's (Android 16) system liblog.so needs.
        pb.environment()["LD_LIBRARY_PATH"] = "$libs:/system/lib64:/vendor/lib64:/system/vendor/lib64"
        pb.environment()["ADSP_LIBRARY_PATH"] = libs
        val slots = expertSlots(ctx)
        if (slots > 0) {
            pb.environment()["GGML_HEXAGON_EXPERT_SLOTS"] = slots.toString()
            pb.environment()["GGML_HEXAGON_EXPERT_FILE"] = expertFile(ctx).absolutePath
            // Low memory stops for the chip once per layer to see which experts it picked; watching for the answer
            // instead of sleeping until woken wakes it far sooner: 14.8 -> 18.6 words/s on the S25 Edge (2026-10-10).
            // Costs one busy core while Ling writes. The same switch gave Fastest only ~2%, so it stays off there.
            pb.environment()["GGML_HEXAGON_OPPOLL"] = "1"
            if (!File(expertFile(ctx).path + ".done").isFile) say("Preparing Ling for low memory (first time, about a minute)…")
        }
        pb.redirectErrorStream(true)
        val started = try { pb.start() } catch (e: Exception) {
            keep("could not start: ${e.message}"); say("Could not start ${name(ctx)}: ${e.message}"); return false
        }
        proc = started
        Thread {
            runCatching {
                started.inputStream.bufferedReader().forEachLine { line ->
                    if (line.isNotBlank()) { keep(line); Log.i("ling", line.take(300)) }
                }
            }
        }.apply { isDaemon = true }.start()
        say("Loading ${name(ctx)}…")
        val until = System.currentTimeMillis() + 180_000
        while (System.currentTimeMillis() < until) {
            if (!started.isAlive) {
                proc = null
                say("${name(ctx)} stopped while loading: $lastLog")
                return false
            }
            if (answers()) { Log.i("ling", "ready"); return true }
            Thread.sleep(300)
        }
        say("${name(ctx)} did not become ready: $healthError / $lastLog")
        stop()
        return false
    }

    /** Stops the server and gives its memory back (the image engines may need it on a 12 GB phone). */
    @Synchronized
    fun stop() {
        val p = proc ?: return
        proc = null
        runCatching { p.destroy() }
        runCatching { p.waitFor() }
    }

    /** Whether it is up and answering yet (503 while the model loads). */
    private fun answers(): Boolean = try {
        val c = URL("$BASE_URL/health").openConnection() as HttpURLConnection
        c.connectTimeout = 500
        c.readTimeout = 1000
        val code = c.responseCode
        c.disconnect()
        code in 200..299
    } catch (e: Exception) {
        healthError = e.toString()
        false
    }

    /** Why the last readiness check failed (shown if it never comes up: the iQOO refused plain HTTP to itself). */
    @Volatile private var healthError = ""
}
