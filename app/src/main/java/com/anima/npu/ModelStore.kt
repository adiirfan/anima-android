package com.anima.npu

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import kotlinx.coroutines.delay
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * Downloads the model files from Hugging Face into files/models.
 *
 * The list of files (name, size, sha256) comes from manifest.json in the model repo, so the files can be
 * updated without a new app. Downloads go through Android's DownloadManager: it resumes after network
 * drops, keeps going while the app is in the background, and runs in its own process (opening the
 * neural unit costs our process its hostname lookups, see the Mihon notes). Each file lands as
 * <name>.part, is checked against the manifest, then renamed, so a broken file never reaches the engine.
 */
class ModelStore(
    val id: String, val title: String, val licence: String,
    private val eliteUrl: String,   // the file list for Snapdragon 8 Elite (Hexagon v79)
    private val gen5Url: String?,   // the same for 8 Elite Gen 5 (Hexagon v81); null when not built for it yet
    private val sub: String,  // folder under files/models ("" for Anima, which predates the others)
    private val gpuUrl: String? = null,   // a graphics-chip build (the app's own engine, any phone), if any
    private val onGpu: Boolean = false,   // this store is the graphics-chip build of the model [baseId]
    val baseId: String = id,
    /** The same downloaded files run on either chip (the chat models: llama.cpp reads them on the AI chip or the
     *  graphics chip), so there is no separate graphics-chip download. */
    val sameFilesOnGpu: Boolean = false,
    /** The file lists for the Snapdragon 8 Gen 3 (v75) and 8 Gen 2 (v73), for the models built for them (Anima). */
    private val gen3Url: String? = null,
    private val gen2Url: String? = null,
) {
    /**
     * Where this build's file list is, or null when there is none for this phone: the AI-chip build of the phone's
     * Snapdragon, or the graphics-chip build (one for every phone).
     */
    val manifestUrl: String?
        get() = if (onGpu) gpuUrl else when (Chip.kind) {
            Chip.Kind.ELITE -> eliteUrl
            Chip.Kind.ELITE_GEN5 -> gen5Url
            // the chat models' llama.cpp has no AI-chip support for these chips: their files run on the graphics chip
            Chip.Kind.GEN3 -> if (sameFilesOnGpu) eliteUrl else gen3Url
            Chip.Kind.GEN2 -> if (sameFilesOnGpu) eliteUrl else gen2Url
            Chip.Kind.UNSUPPORTED -> if (sameFilesOnGpu) eliteUrl else null
        }

    /** This store is a graphics-chip build. */
    val onGraphicsChip get() = onGpu

    /** Whether this phone can download this build. */
    val available get() = manifestUrl != null
    /** This build's Hugging Face page (the repo its files come from). */
    val pageUrl: String? get() = manifestUrl?.substringBefore("/resolve/")

    /** The model's graphics-chip build, in models/<model>/gpu/, or null when the model has none. */
    val gpu: ModelStore? by lazy {
        if (onGpu || gpuUrl == null) null
        else ModelStore("$id-gpu", title, licence, eliteUrl, gen5Url, "${sub.ifEmpty { "anima/" }}gpu/", gpuUrl, true, id)
    }

    /** The AI-chip build exists for this phone (shown as a choice only then). */
    val aiChip get() = !onGpu && manifestUrl != null && !(sameFilesOnGpu && !Chip.supported)

    /** This phone can run the model one way or the other. */
    val offered get() = available || gpu?.available == true

    private fun prefs(ctx: Context) = ctx.getSharedPreferences("anima", Context.MODE_PRIVATE)

    /**
     * The model draws on the graphics chip: the user picked it on the models page, or there is no AI-chip build
     * for this phone. Until the user picks, a build that is already downloaded wins, the AI chip first.
     */
    fun runsOnGpu(ctx: Context): Boolean {
        if (sameFilesOnGpu) return !aiChip || prefs(ctx).getString("runtime_$id", null) == "gpu"
        val g = gpu ?: return false
        if (!aiChip) return true
        prefs(ctx).getString("runtime_$id", null)?.let { return it == "gpu" && g.available }
        return installed(ctx) == null && g.installed(ctx) != null
    }

    fun setRunsOnGpu(ctx: Context, on: Boolean) {
        prefs(ctx).edit().putString("runtime_$id", if (on) "gpu" else "npu").apply()
    }

    /** The build that draws: this one or its graphics-chip twin. */
    fun active(ctx: Context): ModelStore = if (runsOnGpu(ctx)) gpu ?: this else this   // chat models: same files, no twin

    companion object {
        private const val HF = "https://huggingface.co/sebekblessing"
        // the chat models are plain llama.cpp files that run on either chip: one repo of their own
        private const val CHAT = "$HF/anima-chat-models/resolve/main"
        val ANIMA = ModelStore(
            "anima", "Anima",
            "Anima is licensed for non-commercial use (CircleStone Labs Non-Commercial License). Pictures you make may be used for anything.",
            // multi-size + guide set; manifest.json stays for older app builds
            "$HF/anima-qnn-8elite/resolve/main/manifest2.json", "$HF/anima-qnn-8elite-gen5/resolve/main/manifest2.json", "",
            // manifest2: the drawing file in scale blocks of 256 (the Vulkan engine's fast layout); manifest.json keeps
            // the blocks-of-64 file for older app builds, whose graphics engine reads only that
            "$HF/anima-gpu/resolve/main/manifest2.json",
            gen3Url = "$HF/anima-qnn-8gen3/resolve/main/manifest2.json", gen2Url = "$HF/anima-qnn-8gen2/resolve/main/manifest2.json")
        val KLEIN = ModelStore(
            "klein", "FLUX.2 klein 4B",
            "FLUX.2 [klein] 4B is released by Black Forest Labs under the Apache 2.0 licence.",
            "$HF/klein-4b-qnn-8elite/resolve/main/manifest.json", "$HF/klein-4b-qnn-8elite-gen5/resolve/main/manifest.json", "klein/",
            "$HF/klein-4b-gpu/resolve/main/manifest2.json")   // manifest2: the Vulkan layout (256-input blocks), as Anima
        val QWEN = ModelStore(
            "qwen", "Qwen-Image 2.1",
            "Qwen-Image 2.1 is released under the Qwen Research License: non-commercial use only. Built with Qwen.",
            "$HF/qwen-image-2.1-qnn-8elite/resolve/main/manifest.json", "$HF/qwen-image-2.1-qnn-8elite-gen5/resolve/main/manifest.json", "qwen/",
            "$HF/qwen-image-2.1-gpu/resolve/main/manifest2.json")
        val FIBO = ModelStore(
            "fibo", "Fibo 2.0 edit",
            "Fibo 2.0 is released by Bria under CC BY-NC 4.0: non-commercial use only.",
            "$HF/fibo-qnn-8elite/resolve/main/manifest.json", "$HF/fibo-qnn-8elite-gen5/resolve/main/manifest.json", "fibo/",
            "$HF/fibo-gpu/resolve/main/manifest2.json")
        val MOSS = ModelStore(
            "moss", "MOSS-Image",
            "MOSS-Image and its 8-step add-on are released under the Apache 2.0 licence.",
            "$HF/moss-qnn-8elite/resolve/main/manifest.json", "$HF/moss-qnn-8elite-gen5/resolve/main/manifest.json", "moss/",
            "$HF/moss-gpu/resolve/main/manifest2.json")   // manifest2: the Vulkan layout (256-input blocks), as Anima
        val WAN = ModelStore(
            "wan", "Wan 2.2 clips",
            "Wan 2.2 (Alibaba), its LightTAE decoder and the umT5 prompt reader are released under the Apache 2.0 " +
                "licence. The Turbo version's page names no licence.",
            "$HF/wan22-qnn-8elite/resolve/main/manifest.json", "$HF/wan22-qnn-8elite-gen5/resolve/main/manifest.json", "wan/",
            "$HF/wan22-gpu/resolve/main/manifest2.json")
        val LING = ModelStore(
            "ling", "Ling-3.0-tiny (chat)",
            "Ling-3.0-tiny is released by inclusionAI under the MIT licence.",
            "$CHAT/ling/manifest.json", "$CHAT/ling/manifest.json", "ling/", sameFilesOnGpu = true)
        val SMALLCHAT = ModelStore(
            "smallchat", "Qwen3.5 0.8B (small chat)",
            "Qwen3.5 0.8B is released by the Qwen team (Alibaba) under the Apache 2.0 licence.",
            "$CHAT/smallchat/manifest.json", "$CHAT/smallchat/manifest.json", "smallchat/", sameFilesOnGpu = true)
        val ALL = listOf(ANIMA, KLEIN, QWEN, FIBO, MOSS, WAN, LING, SMALLCHAT)
        /** The picture models, for the Create screen's picker. */
        val PICTURE = listOf(ANIMA, KLEIN, QWEN, FIBO, MOSS)
        /** The first-run setup's lists: what makes pictures (and clips), and the chat models. */
        val SETUP_PICTURE = listOf(ANIMA, MOSS, FIBO, WAN, QWEN, KLEIN)
        val SETUP_CHAT = listOf(LING, SMALLCHAT)
        fun byId(id: String) = ALL.firstOrNull { it.id == id } ?: ALL.mapNotNull { it.gpu }.first { it.id == id }
    }

    data class FileInfo(val path: String, val name: String, val size: Long, val sha256: String)
    data class Variant(val id: String, val label: String, val about: String, val files: List<FileInfo>)
    /**
     * A model's file list. [version] goes up whenever its files change on Hugging Face, and [notes] says what
     * changed; an update downloads only the files whose checksum changed, never the whole model again.
     */
    data class Manifest(
        val base: String, val shared: List<FileInfo>, val variants: List<Variant>, val version: Int = 1, val notes: String = "",
    ) {
        fun variant(id: String) = variants.first { it.id == id }
        fun bytes(id: String) = (shared + variant(id).files).sumOf { it.size }
    }

    fun modelDir(ctx: Context) = File(ctx.getExternalFilesDir(null), "models/$sub")
    private fun manifestFile(ctx: Context) = File(modelDir(ctx), "manifest.json")
    private fun variantFile(ctx: Context) = File(modelDir(ctx), "variant.txt")
    // name -> sha256 of every file as installed, so a changed manifest shows up as an update
    private fun installedFile(ctx: Context) = File(modelDir(ctx), "installed.json")
    // the manifest version the installed files came from (absent = 1, the lists before versions existed)
    private fun versionFile(ctx: Context) = File(modelDir(ctx), "version.txt")

    /** The installed files' version number. */
    fun installedVersion(ctx: Context): Int = versionFile(ctx).takeIf { it.isFile }?.readText()?.trim()?.toIntOrNull() ?: 1

    /** When every installed file already matches [m] (nothing to update), the installed files are [m]'s version. */
    fun noteVersion(ctx: Context, m: Manifest) {
        if (installed(ctx) != null && m.version > installedVersion(ctx) && outdated(ctx, m).isEmpty())
            versionFile(ctx).writeText(m.version.toString())
    }

    private fun installedShas(ctx: Context): Map<String, String> =
        installedFile(ctx).takeIf { it.isFile }?.let { f ->
            runCatching { JSONObject(f.readText()).let { o -> o.keys().asSequence().associateWith { o.getString(it) } } }.getOrNull()
        } ?: emptyMap()

    /** Files of the installed variant that differ from [m] (new model files on Hugging Face). */
    fun outdated(ctx: Context, m: Manifest): List<FileInfo> {
        val id = installed(ctx) ?: return emptyList()
        if (m.variants.none { it.id == id }) return emptyList()
        val have = installedShas(ctx)
        val dir = modelDir(ctx)
        return (m.shared + m.variant(id).files).filter { f -> File(dir, f.name).length() != f.size && have[f.name] == null || have[f.name] != null && have[f.name] != f.sha256 }
    }

    fun cachedManifest(ctx: Context): Manifest? =
        manifestFile(ctx).takeIf { it.isFile }?.let { runCatching { parse(it.readText()) }.getOrNull() }

    private fun parse(s: String): Manifest {
        val o = JSONObject(s)
        fun files(a: org.json.JSONArray) = List(a.length()) { i ->
            val f = a.getJSONObject(i)
            FileInfo(f.getString("path"), f.getString("name"), f.getLong("size"), f.getString("sha256"))
        }
        val v = o.getJSONObject("variants")
        val variants = v.keys().asSequence().map { k ->
            val x = v.getJSONObject(k)
            Variant(k, x.getString("label"), x.getString("about"), files(x.getJSONArray("files")))
        }.toList()
        return Manifest(o.getString("base"), files(o.getJSONArray("shared")), variants, o.optInt("version", 1), o.optString("notes", ""))
    }

    /**
     * The installed variant id, or null when nothing complete is installed. Judged by the record of what was
     * installed (installed.json), not by the current manifest, so newer files on Hugging Face show up as an
     * update instead of making the installed model look broken.
     */
    fun installed(ctx: Context): String? {
        val id = variantFile(ctx).takeIf { it.isFile }?.readText()?.trim() ?: return null
        val dir = modelDir(ctx)
        val record = installedShas(ctx)
        if (record.isNotEmpty()) return if (record.keys.all { File(dir, it).isFile }) id else null
        // Installed before the record existed: accept it if it matches the cached manifest, and write the record.
        val m = cachedManifest(ctx) ?: return null
        if (m.variants.none { it.id == id }) return null
        val files = m.shared + m.variant(id).files
        if (!files.all { File(dir, it.name).length() == it.size }) return null
        installedFile(ctx).writeText(JSONObject(files.associate { it.name to it.sha256 }).toString())
        return id
    }

    /**
     * Removes this build's files (not the other build's folder inside it). The caller frees the engines first.
     * The cached file list stays, so the page can still say how big a new download would be.
     */
    fun delete(ctx: Context) {
        variantFile(ctx).delete()   // first: a half-deleted model must never look installed
        modelDir(ctx).listFiles()?.filter { it.isFile && it.name != "manifest.json" }?.forEach { it.delete() }
    }

    /**
     * Phones without the AI chip used to get the graphics-chip build straight into models/<model>/. It now lives
     * in models/<model>/gpu/ next to the AI-chip build; moves the files there once (a rename, no copying).
     */
    fun moveOldGpuInstall(ctx: Context) {
        val g = gpu ?: return
        val dir = modelDir(ctx)
        if (variantFile(ctx).takeIf { it.isFile }?.readText()?.trim() != "gpu") return
        val to = g.modelDir(ctx).apply { mkdirs() }
        dir.listFiles()?.filter { it.isFile }?.forEach { it.renameTo(File(to, it.name)) }
    }

    private fun dm(ctx: Context) = ctx.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager

    private fun enqueue(ctx: Context, url: String, name: String, title: String): Long {
        File(modelDir(ctx), name).delete()
        val req = DownloadManager.Request(Uri.parse(url))
            .setTitle(title)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setDestinationInExternalFilesDir(ctx, null, "models/$sub$name")
        return dm(ctx).enqueue(req)
    }

    /** (bytes so far, status) for each id; status is a DownloadManager.STATUS_* value. */
    private fun query(ctx: Context, ids: List<Long>): Map<Long, Pair<Long, Int>> {
        val out = HashMap<Long, Pair<Long, Int>>()
        if (ids.isEmpty()) return out
        dm(ctx).query(DownloadManager.Query().setFilterById(*ids.toLongArray())).use { c ->
            val idCol = c.getColumnIndexOrThrow(DownloadManager.COLUMN_ID)
            val soFar = c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
            val st = c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)
            while (c.moveToNext()) out[c.getLong(idCol)] = c.getLong(soFar) to c.getInt(st)
        }
        return out
    }

    /** This app's transfers in Android's download service that have not failed: (id, address, status). */
    private fun ours(ctx: Context): List<Triple<Long, String, Int>> {
        val out = ArrayList<Triple<Long, String, Int>>()
        dm(ctx).query(DownloadManager.Query().setFilterByStatus(DownloadManager.STATUS_PENDING or DownloadManager.STATUS_RUNNING or
            DownloadManager.STATUS_PAUSED or DownloadManager.STATUS_SUCCESSFUL)).use { c ->
            val idCol = c.getColumnIndexOrThrow(DownloadManager.COLUMN_ID)
            val uCol = c.getColumnIndexOrThrow(DownloadManager.COLUMN_URI)
            val sCol = c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)
            while (c.moveToNext()) out += Triple(c.getLong(idCol), c.getString(uCol) ?: "", c.getInt(sCol))
        }
        return out
    }

    /** Where Android saved download [id], if it says. */
    private fun localFile(ctx: Context, id: Long): File? =
        dm(ctx).query(DownloadManager.Query().setFilterById(id)).use { c ->
            if (!c.moveToFirst()) return null
            val u = c.getString(c.getColumnIndexOrThrow(DownloadManager.COLUMN_LOCAL_URI)) ?: return null
            Uri.parse(u).path?.let { File(it) }?.takeIf { it.isFile }
        }

    private suspend fun waitFor(ctx: Context, ids: List<Long>, onBytes: (Long) -> Unit) {
        while (true) {
            val q = query(ctx, ids)
            if (q.size < ids.size) error("download cancelled")
            q.values.firstOrNull { it.second == DownloadManager.STATUS_FAILED }?.let { error("a download failed — check the connection and try again") }
            onBytes(q.values.sumOf { it.first })
            if (q.values.all { it.second == DownloadManager.STATUS_SUCCESSFUL }) return
            delay(500)
        }
    }

    /** Fetches the current manifest from Hugging Face (through DownloadManager) and caches it. */
    suspend fun fetchManifest(ctx: Context): Manifest {
        modelDir(ctx).mkdirs()
        val url = manifestUrl ?: error("$title is not available for ${Chip.label}")
        // strays from checks that overlapped: Android saves under "manifest.json-1.part" when the name is taken
        modelDir(ctx).listFiles()?.filter { Regex("manifest\\.json-\\d+\\.part").matches(it.name) }?.forEach { it.delete() }
        val id = enqueue(ctx, url, "manifest.json.part", "$title model list")
        waitFor(ctx, listOf(id)) {}
        // the file Android actually wrote (another check of the same model may have held the usual name)
        val part = localFile(ctx, id) ?: File(modelDir(ctx), "manifest.json.part")
        val m = parse(part.readText())
        part.renameTo(manifestFile(ctx))
        return m
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s ->
            val buf = ByteArray(1 shl 20)
            while (true) { val n = s.read(buf); if (n < 0) break; md.update(buf, 0, n) }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    /**
     * Downloads whatever [variant] still needs (files already present with the right size are kept),
     * checks every new file, then marks the variant installed and removes the other variant's leftovers.
     * [progress] gets (done bytes, total bytes, what is happening).
     */
    suspend fun install(ctx: Context, m: Manifest, variant: String, progress: (Long, Long, String) -> Unit) {
        val dir = modelDir(ctx).apply { mkdirs() }
        val switching = variantFile(ctx).takeIf { it.isFile }?.readText()?.trim() != variant
        val wanted = m.shared + m.variant(variant).files
        val shas = installedShas(ctx)
        // A file is kept only if it has the right size and (when its checksum is on record) the right checksum.
        // Drawing pieces of different variants share names, so they are never kept when switching.
        val need = wanted.filter { f ->
            val have = File(dir, f.name)
            val sameSha = shas[f.name]?.let { it == f.sha256 } ?: true
            !(have.length() == f.size && sameSha && (!switching || f in m.shared))
        }
        val total = need.sumOf { it.size }
        // Android's download service keeps going when the app is closed or updated: a transfer of the same file that
        // is still alive (started before the app restarted, by this build or an older one) is picked up, not redone
        val live = ours(ctx)
        val ids = need.map { f ->
            val url = m.base + f.path
            val part = File(dir, f.name + ".part")
            live.firstOrNull { (_, u, st) -> u == url && (st != DownloadManager.STATUS_SUCCESSFUL || part.length() == f.size) }?.first
                ?: enqueue(ctx, url, f.name + ".part", "$title: ${f.name}")
        }
        try {
            waitFor(ctx, ids) { progress(it, total, "Downloading") }
        } catch (e: Exception) {
            dm(ctx).remove(*ids.toLongArray())
            throw e
        }
        for ((i, f) in need.withIndex()) {
            progress(total, total, "Checking ${i + 1} of ${need.size}")
            val part = File(dir, f.name + ".part")
            if (part.length() != f.size || sha256(part) != f.sha256) {
                part.delete()
                error("${f.name} arrived damaged — please download again")
            }
        }
        variantFile(ctx).delete()  // half-switched state must not look installed
        for (f in need) File(dir, f.name + ".part").renameTo(File(dir, f.name))
        installedFile(ctx).writeText(JSONObject(wanted.associate { it.name to it.sha256 }).toString())
        versionFile(ctx).writeText(m.version.toString())
        variantFile(ctx).writeText(variant)
        // files the installed set no longer uses (the 16-bit word table once the 8-bit one is in, 311 MB)
        val keep = wanted.map { it.name }.toSet() + setOf("manifest.json", "installed.json", "variant.txt", "version.txt")
        dir.listFiles()?.filter { it.isFile && it.name !in keep && !it.name.endsWith(".part") }?.forEach { it.delete() }
    }
}
