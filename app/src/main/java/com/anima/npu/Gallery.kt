package com.anima.npu

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Size
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Saved pictures live in the phone's Pictures/Anima folder (so the normal gallery app shows them too);
 * the prompt, seed and steps of each are kept in files/gallery.json, keyed by the picture's address.
 */
object Gallery {
    data class Entry(val uri: Uri, val prompt: String, val seed: Long, val steps: Int, val ms: Long, val width: Int, val savedAt: Long, val height: Int = width)

    private fun index(ctx: Context) = File(ctx.getExternalFilesDir(null), "gallery.json")

    private fun read(ctx: Context): MutableList<Entry> {
        val f = index(ctx)
        if (!f.isFile) return mutableListOf()
        val a = JSONArray(f.readText())
        return MutableList(a.length()) { i ->
            val o = a.getJSONObject(i)
            Entry(Uri.parse(o.getString("uri")), o.getString("prompt"), o.getLong("seed"), o.getInt("steps"),
                o.optLong("ms"), o.optInt("width", 512), o.optLong("savedAt"), o.optInt("height", o.optInt("width", 512)))
        }
    }

    private fun write(ctx: Context, list: List<Entry>) {
        val a = JSONArray()
        list.forEach { e ->
            a.put(JSONObject().put("uri", e.uri.toString()).put("prompt", e.prompt).put("seed", e.seed)
                .put("steps", e.steps).put("ms", e.ms).put("width", e.width).put("height", e.height).put("savedAt", e.savedAt))
        }
        index(ctx).writeText(a.toString())
    }

    fun save(ctx: Context, bmp: Bitmap, prompt: String, seed: Long, steps: Int, ms: Long): Entry {
        val now = System.currentTimeMillis()
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "anima_${now}_$seed.png")
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Anima")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val r = ctx.contentResolver
        val uri = r.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("could not create the picture file")
        r.openOutputStream(uri)!!.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        r.update(uri, values, null, null)
        val e = Entry(uri, prompt, seed, steps, ms, bmp.width, now, bmp.height)
        write(ctx, read(ctx).apply { add(0, e) })
        return e
    }

    /** Newest first; entries whose picture was deleted elsewhere (e.g. in the phone's gallery) are dropped. */
    fun list(ctx: Context): List<Entry> {
        val all = read(ctx)
        val alive = all.filter { e ->
            runCatching { ctx.contentResolver.openFileDescriptor(e.uri, "r")?.close(); true }.getOrDefault(false)
        }
        if (alive.size != all.size) write(ctx, alive)
        return alive.sortedByDescending { it.savedAt }
    }

    fun delete(ctx: Context, e: Entry) {
        runCatching { ctx.contentResolver.delete(e.uri, null, null) }
        write(ctx, read(ctx).filter { it.uri != e.uri })
    }

    fun thumbnail(ctx: Context, e: Entry, px: Int): Bitmap? =
        runCatching { ctx.contentResolver.loadThumbnail(e.uri, Size(px, px), null) }.getOrNull()

    fun full(ctx: Context, e: Entry): Bitmap? =
        runCatching { ctx.contentResolver.openInputStream(e.uri)!!.use { android.graphics.BitmapFactory.decodeStream(it) } }.getOrNull()
}
