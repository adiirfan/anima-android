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
    val id: String, val title: String, val licence: String, private val manifestUrl: String,
    private val sub: String,  // folder under files/models ("" for Anima, which predates the others)
) {
    companion object {
        val ANIMA = ModelStore(
            "anima", "Anima",
            "Anima is licensed for non-commercial use (CircleStone Labs Non-Commercial License). Pictures you make may be used for anything.",
            // multi-size + guide set; manifest.json stays for older app builds
            "https://huggingface.co/sebekblessing/anima-qnn-8elite/resolve/main/manifest2.json", "")
        val KLEIN = ModelStore(
            "klein", "FLUX.2 klein 4B",
            "FLUX.2 [klein] 4B is released by Black Forest Labs under the Apache 2.0 licence.",
            "https://huggingface.co/sebekblessing/klein-qnn-8elite/resolve/main/manifest.json", "klein/")
        val QWEN = ModelStore(
            "qwen", "Qwen-Image 2.1",
            "Qwen-Image 2.1 is released under the Qwen Research License: non-commercial use only. Built with Qwen.",
            "https://huggingface.co/sebekblessing/image21-qnn-8elite/resolve/main/manifest.json", "qwen/")
        val LING = ModelStore(
            "ling", "Ling-3.0-tiny (chat)",
            "Ling-3.0-tiny is released by inclusionAI under the MIT licence.",
            "https://huggingface.co/sebekblessing/anima-qnn-8elite/resolve/main/ling/manifest.json", "ling/")
        val SMALLCHAT = ModelStore(
            "smallchat", "Qwen3.5 0.8B (small chat)",
            "Qwen3.5 0.8B is released by the Qwen team (Alibaba) under the Apache 2.0 licence.",
            "https://huggingface.co/sebekblessing/anima-qnn-8elite/resolve/main/smallchat/manifest.json", "smallchat/")
        val ALL = listOf(ANIMA, KLEIN, QWEN, LING, SMALLCHAT)
        /** The picture models, for the Create screen's picker. */
        val PICTURE = listOf(ANIMA, KLEIN, QWEN)
        fun byId(id: String) = ALL.first { it.id == id }
    }

    data class FileInfo(val path: String, val name: String, val size: Long, val sha256: String)
    data class Variant(val id: String, val label: String, val about: String, val files: List<FileInfo>)
    data class Manifest(val base: String, val shared: List<FileInfo>, val variants: List<Variant>) {
        fun variant(id: String) = variants.first { it.id == id }
        fun bytes(id: String) = (shared + variant(id).files).sumOf { it.size }
    }

    fun modelDir(ctx: Context) = File(ctx.getExternalFilesDir(null), "models/$sub")
    private fun manifestFile(ctx: Context) = File(modelDir(ctx), "manifest.json")
    private fun variantFile(ctx: Context) = File(modelDir(ctx), "variant.txt")
    // name -> sha256 of every file as installed, so a changed manifest shows up as an update
    private fun installedFile(ctx: Context) = File(modelDir(ctx), "installed.json")

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
        return Manifest(o.getString("base"), files(o.getJSONArray("shared")), variants)
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
        val id = enqueue(ctx, manifestUrl, "manifest.json.part", "$title model list")
        waitFor(ctx, listOf(id)) {}
        val part = File(modelDir(ctx), "manifest.json.part")
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
        val ids = need.map { enqueue(ctx, m.base + it.path, it.name + ".part", "$title: ${it.name}") }
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
        variantFile(ctx).writeText(variant)
        // files the installed set no longer uses (the 16-bit word table once the 8-bit one is in, 311 MB)
        val keep = wanted.map { it.name }.toSet() + setOf("manifest.json", "installed.json", "variant.txt")
        dir.listFiles()?.filter { it.isFile && it.name !in keep && !it.name.endsWith(".part") }?.forEach { it.delete() }
    }

    fun cancelAll(ctx: Context) {
        val c = dm(ctx).query(DownloadManager.Query().setFilterByStatus(
            DownloadManager.STATUS_PENDING or DownloadManager.STATUS_RUNNING or DownloadManager.STATUS_PAUSED))
        val ids = ArrayList<Long>()
        c.use { while (it.moveToNext()) ids += it.getLong(it.getColumnIndexOrThrow(DownloadManager.COLUMN_ID)) }
        if (ids.isNotEmpty()) dm(ctx).remove(*ids.toLongArray())
    }
}
