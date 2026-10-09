package com.anima.npu

import android.app.DownloadManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File
import java.security.MessageDigest

/**
 * The app updates itself from the newest release of the public GitHub repo adiirfan/anima-android (anima.apk +
 * latest.json, published by tools/publish-app.sh; "latest/download" always points at the newest release's files).
 * Fetches go through DownloadManager, which follows GitHub's redirect to its file host: once the AI chip has been
 * opened this process can no longer look up host names.
 */
object AppUpdate {
    private const val BASE = "https://github.com/adiirfan/anima-android/releases/latest/download/"
    var available by mutableStateOf<UpdateInfo?>(null)
        private set
    var status by mutableStateOf("")
        private set
    var busy by mutableStateOf(false)
        private set

    private fun dir(ctx: Context) = File(ctx.getExternalFilesDir(null), "updates").apply { mkdirs() }
    private fun dm(ctx: Context) = ctx.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager

    private suspend fun fetch(ctx: Context, name: String, visible: Boolean, onBytes: (Long) -> Unit = {}): File {
        val out = File(dir(ctx), name).apply { delete() }
        val req = DownloadManager.Request(Uri.parse(BASE + name + "?t=" + System.currentTimeMillis()))
            .setNotificationVisibility(if (visible) DownloadManager.Request.VISIBILITY_VISIBLE else DownloadManager.Request.VISIBILITY_HIDDEN)
            .setTitle("Anima update")
            .setDestinationInExternalFilesDir(ctx, null, "updates/$name")
        val id = dm(ctx).enqueue(req)
        while (true) {
            dm(ctx).query(DownloadManager.Query().setFilterById(id)).use { c ->
                if (!c.moveToFirst()) error("download cancelled")
                onBytes(c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)))
                when (c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))) {
                    DownloadManager.STATUS_SUCCESSFUL -> return out
                    DownloadManager.STATUS_FAILED -> { dm(ctx).remove(id); error("download failed") }
                }
            }
            delay(400)
        }
    }

    /** Quietly looks for a newer build; network trouble just leaves [available] as it was. */
    suspend fun check(ctx: Context) = withContext(Dispatchers.IO) {
        if (busy) return@withContext
        runCatching {
            val u = UpdateInfo.parse(fetch(ctx, "latest.json", visible = false).readText())
            available = u.takeIf { it.isNewerThan(BuildConfig.APP_BUILD) }
            android.util.Log.i("anima", "update check: latest ${u.build}, this ${BuildConfig.APP_BUILD}")
        }.onFailure { android.util.Log.w("anima", "update check failed: $it") }
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s -> val b = ByteArray(1 shl 20); while (true) { val n = s.read(b); if (n < 0) break; md.update(b, 0, n) } }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    /** Downloads, checks and opens Android's installer. [blockedBy] names work an install would kill, or null. */
    suspend fun install(ctx: Context, blockedBy: () -> String?) = withContext(Dispatchers.IO) {
        val u = available ?: return@withContext
        blockedBy()?.let { status = "$it: update when it has finished"; return@withContext }
        busy = true
        try {
            status = "Downloading the update…"
            val apk = fetch(ctx, "anima.apk", visible = true) { b -> status = "Downloading the update… %.0f of %.0f MB".format(b / 1e6, u.size / 1e6) }
            status = "Checking…"
            if (apk.length() != u.size || sha256(apk) != u.sha256) { apk.delete(); error("the update arrived damaged") }
            status = ""
            val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".updates", apk)
            ctx.startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(uri, "application/vnd.android.package-archive")
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK))
        } catch (e: Exception) {
            status = "Update failed: ${e.message}"
        } finally {
            busy = false
        }
    }
}
