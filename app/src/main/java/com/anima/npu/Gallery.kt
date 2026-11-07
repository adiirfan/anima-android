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

    /** A clip as a looping GIF in Pictures/Anima; [frames]: w*h ARGB each. The entry's size is one frame's. */
    fun saveGif(ctx: Context, frames: List<IntArray>, w: Int, h: Int, prompt: String, seed: Long, steps: Int, ms: Long): Entry {
        val now = System.currentTimeMillis()
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "anima_${now}_$seed.gif")
            put(MediaStore.Images.Media.MIME_TYPE, "image/gif")
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Anima")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val r = ctx.contentResolver
        val uri = r.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("could not create the clip file")
        r.openOutputStream(uri)!!.use { GifWriter.write(it, frames, w, h) }
        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        r.update(uri, values, null, null)
        val e = Entry(uri, prompt, seed, steps, ms, w, now, h)
        write(ctx, read(ctx).apply { add(0, e) })
        return e
    }

    /** A clip as an MP4 video in Pictures/Anima ([VideoWriter]); [frames]: w*h ARGB each. The entry's size is a frame's. */
    fun saveVideo(ctx: Context, frames: List<IntArray>, w: Int, h: Int, prompt: String, seed: Long, steps: Int, ms: Long): Entry {
        val now = System.currentTimeMillis()
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, "anima_${now}_$seed.mp4")
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Anima")
            put(MediaStore.Video.Media.IS_PENDING, 1)
        }
        val r = ctx.contentResolver
        val uri = r.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values) ?: error("could not create the clip file")
        try {
            r.openFileDescriptor(uri, "rw")!!.use { VideoWriter.write(it.fileDescriptor, frames, w, h) }
        } catch (t: Throwable) {
            runCatching { r.delete(uri, null, null) }; throw t
        }
        values.clear()
        values.put(MediaStore.Video.Media.IS_PENDING, 0)
        r.update(uri, values, null, null)
        val e = Entry(uri, prompt, seed, steps, ms, w, now, h)
        write(ctx, read(ctx).apply { add(0, e) })
        return e
    }

    /** Puts [bmp] in place of [old]: a new file with the same prompt, seed and place in the list; the old file is
     *  deleted, and queue jobs that pointed at it point at the new one. */
    fun replace(ctx: Context, old: Entry, bmp: Bitmap): Entry {
        val e = save(ctx, bmp, old.prompt, old.seed, old.steps, old.ms).copy(savedAt = old.savedAt)
        runCatching { ctx.contentResolver.delete(old.uri, null, null) }
        write(ctx, read(ctx).filter { it.uri != old.uri && it.uri != e.uri }.toMutableList().apply { add(0, e) })
        Queue.store.all().filter { it.resultUri == old.uri.toString() }.forEach { j ->
            Queue.store.update(j.id) { it.copy(resultUri = e.uri.toString(), width = bmp.width, height = bmp.height) }
        }
        return e
    }

    fun mime(ctx: Context, e: Entry): String = runCatching { ctx.contentResolver.getType(e.uri) }.getOrNull() ?: "image/png"
    fun isGif(ctx: Context, e: Entry) = mime(ctx, e) == "image/gif"
    fun isVideo(ctx: Context, e: Entry) = mime(ctx, e).startsWith("video/")
    /** A moving picture: a Wan clip, MP4 now, GIF before 2026-11-07. */
    fun isClip(ctx: Context, e: Entry) = mime(ctx, e).let { it == "image/gif" || it.startsWith("video/") }

    /** Newest first; entries whose picture was deleted elsewhere (e.g. in the phone's gallery) are dropped.
     *  One query of the phone's picture list instead of opening every file (192 opens took seconds). */
    fun list(ctx: Context): List<Entry> {
        val all = read(ctx)
        val ids = HashSet<String>()
        val asked = runCatching {
            ctx.contentResolver.query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, arrayOf(MediaStore.Images.Media._ID),
                "${MediaStore.Images.Media.RELATIVE_PATH} LIKE ?", arrayOf(Environment.DIRECTORY_PICTURES + "/Anima%"), null)?.use { c ->
                while (c.moveToNext()) ids.add(c.getLong(0).toString())
            } != null
        }.getOrDefault(false)
        val media = MediaStore.Images.Media.EXTERNAL_CONTENT_URI.toString()
        val alive = all.filter { e ->
            if (asked && e.uri.toString().startsWith(media)) e.uri.lastPathSegment in ids
            else runCatching { ctx.contentResolver.openFileDescriptor(e.uri, "r")?.close(); true }.getOrDefault(false)
        }
        if (alive.size != all.size) write(ctx, alive)
        return alive.sortedByDescending { it.savedAt }
    }

    fun delete(ctx: Context, e: Entry) {
        runCatching { ctx.contentResolver.delete(e.uri, null, null) }
        write(ctx, read(ctx).filter { it.uri != e.uri })
    }

    // Small previews already made, so reopening the gallery shows them at once (about 48 MB at most).
    private val thumbs = object : android.util.LruCache<String, Bitmap>(48 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap) = value.allocationByteCount
    }

    fun cachedThumbnail(e: Entry, px: Int): Bitmap? = thumbs.get("${e.uri}@$px")

    fun thumbnail(ctx: Context, e: Entry, px: Int): Bitmap? {
        cachedThumbnail(e, px)?.let { return it }
        return runCatching { ctx.contentResolver.loadThumbnail(e.uri, Size(px, px), null) }.getOrNull()
            ?.also { thumbs.put("${e.uri}@$px", it) }
    }

    /** The picture for viewing: a 3072-pixel upscale is read at half or quarter size (no screen shows more, and the
     *  whole one is 36 MB in memory and slow to read). */
    fun full(ctx: Context, e: Entry, maxSide: Int = 2048): Bitmap? = if (isVideo(ctx, e)) runCatching {
        android.media.MediaMetadataRetriever().run {
            try { setDataSource(ctx, e.uri); getFrameAtTime(0) } finally { release() }
        }
    }.getOrNull() else runCatching {
        android.graphics.ImageDecoder.decodeBitmap(android.graphics.ImageDecoder.createSource(ctx.contentResolver, e.uri)) { d, info, _ ->
            var s = 1
            while (maxOf(info.size.width, info.size.height) / (s * 2) >= maxSide) s *= 2
            if (maxOf(info.size.width, info.size.height) / s > maxSide) s *= 2
            d.setTargetSampleSize(s)
            d.allocator = android.graphics.ImageDecoder.ALLOCATOR_SOFTWARE
        }
    }.getOrNull()
}
