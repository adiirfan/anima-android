package com.anima.npu

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
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
    /** [kind]: "edit" for a picture made from another picture (an edit or an upscale), else "picture" (also for
     *  pictures saved before 2026-11-24, which did not record it). Clips are told apart by their file type. */
    data class Entry(val uri: Uri, val prompt: String, val seed: Long, val steps: Int, val ms: Long, val width: Int, val savedAt: Long,
                     val height: Int = width, val kind: String = "picture")

    private fun index(ctx: Context) = File(ctx.getExternalFilesDir(null), "gallery.json")

    private fun read(ctx: Context): MutableList<Entry> {
        val f = index(ctx)
        if (!f.isFile) return mutableListOf()
        val a = JSONArray(f.readText())
        return MutableList(a.length()) { i ->
            val o = a.getJSONObject(i)
            Entry(Uri.parse(o.getString("uri")), o.getString("prompt"), o.getLong("seed"), o.getInt("steps"),
                o.optLong("ms"), o.optInt("width", 512), o.optLong("savedAt"), o.optInt("height", o.optInt("width", 512)),
                o.optString("kind", "picture"))
        }
    }

    private fun write(ctx: Context, list: List<Entry>) {
        val a = JSONArray()
        list.forEach { e ->
            a.put(JSONObject().put("uri", e.uri.toString()).put("prompt", e.prompt).put("seed", e.seed)
                .put("steps", e.steps).put("ms", e.ms).put("width", e.width).put("height", e.height).put("savedAt", e.savedAt).put("kind", e.kind))
        }
        index(ctx).writeText(a.toString())
    }

    /** Android 9 and older have no RELATIVE_PATH / IS_PENDING: the file goes straight into Pictures/Anima (with the
     *  storage permission asked for at start) and is registered by its path. */
    private val legacy get() = Build.VERSION.SDK_INT < 29

    /** A new, empty entry in the phone's picture list, in Pictures/Anima. */
    private fun create(ctx: Context, collection: Uri, name: String, mime: String, what: String): Uri {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, mime)
            if (legacy) {
                if (listOf(android.Manifest.permission.READ_EXTERNAL_STORAGE, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                        .any { ctx.checkSelfPermission(it) != android.content.pm.PackageManager.PERMISSION_GRANTED })
                    error("Anima needs storage access to save to your Pictures folder on this Android version: allow Storage for Anima in the phone's Settings, Apps")
                @Suppress("DEPRECATION")
                val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "Anima")
                if (!dir.isDirectory && !dir.mkdirs()) error("could not make the folder $dir")
                @Suppress("DEPRECATION") put(MediaStore.MediaColumns.DATA, File(dir, name).absolutePath)
            } else {
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Anima")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }
        return ctx.contentResolver.insert(collection, values) ?: error("could not create the $what file")
    }

    /** The file is written: show it to other apps (newer Android) or have its size and dimensions read (older). */
    private fun finish(ctx: Context, uri: Uri) {
        if (!legacy) {
            ctx.contentResolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
            return
        }
        @Suppress("DEPRECATION")
        val path = runCatching {
            ctx.contentResolver.query(uri, arrayOf(MediaStore.MediaColumns.DATA), null, null, null)?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
        }.getOrNull() ?: return
        android.media.MediaScannerConnection.scanFile(ctx, arrayOf(path), null, null)
    }

    fun save(ctx: Context, bmp: Bitmap, prompt: String, seed: Long, steps: Int, ms: Long, kind: String = "picture"): Entry {
        val now = System.currentTimeMillis()
        val r = ctx.contentResolver
        val uri = create(ctx, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "anima_${now}_$seed.png", "image/png", "picture")
        r.openOutputStream(uri)!!.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        finish(ctx, uri)
        val e = Entry(uri, prompt, seed, steps, ms, bmp.width, now, bmp.height, kind)
        write(ctx, read(ctx).apply { add(0, e) })
        return e
    }

    /** A clip as a looping GIF in Pictures/Anima; [frames]: w*h ARGB each. The entry's size is one frame's. */
    fun saveGif(ctx: Context, frames: List<IntArray>, w: Int, h: Int, prompt: String, seed: Long, steps: Int, ms: Long): Entry {
        val now = System.currentTimeMillis()
        val r = ctx.contentResolver
        val uri = create(ctx, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "anima_${now}_$seed.gif", "image/gif", "clip")
        r.openOutputStream(uri)!!.use { GifWriter.write(it, frames, w, h) }
        finish(ctx, uri)
        val e = Entry(uri, prompt, seed, steps, ms, w, now, h)
        write(ctx, read(ctx).apply { add(0, e) })
        return e
    }

    /** A clip as an MP4 video in Pictures/Anima ([VideoWriter]); [frames]: w*h ARGB each. The entry's size is a frame's. */
    fun saveVideo(ctx: Context, frames: List<IntArray>, w: Int, h: Int, prompt: String, seed: Long, steps: Int, ms: Long): Entry {
        val now = System.currentTimeMillis()
        val r = ctx.contentResolver
        val uri = create(ctx, MediaStore.Video.Media.EXTERNAL_CONTENT_URI, "anima_${now}_$seed.mp4", "video/mp4", "clip")
        try {
            r.openFileDescriptor(uri, "rw")!!.use {
                // long clips (3 s, 49 frames) play once at 16 per second as made; short ones keep the old back-and-forth loop
                if (frames.size >= 33) VideoWriter.write(it.fileDescriptor, frames, w, h, fps = 16, loops = 1, pingPong = false)
                else VideoWriter.write(it.fileDescriptor, frames, w, h)
            }
        } catch (t: Throwable) {
            runCatching { r.delete(uri, null, null) }; throw t
        }
        finish(ctx, uri)
        val e = Entry(uri, prompt, seed, steps, ms, w, now, h)
        write(ctx, read(ctx).apply { add(0, e) })
        return e
    }

    /** Puts [bmp] in place of [old]: a new file with the same prompt, seed and place in the list; the old file is
     *  deleted, and queue jobs that pointed at it point at the new one. */
    fun replace(ctx: Context, old: Entry, bmp: Bitmap): Entry {
        // the kind as the gallery recorded it: [old] may be rebuilt from a queue job, which does not know it
        val kind = read(ctx).firstOrNull { it.uri == old.uri }?.kind ?: old.kind
        val e = save(ctx, bmp, old.prompt, old.seed, old.steps, old.ms, kind).copy(savedAt = old.savedAt)
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
            @Suppress("DEPRECATION")
            val (where, arg) = if (legacy) "${MediaStore.MediaColumns.DATA} LIKE ?" to "%/" + Environment.DIRECTORY_PICTURES + "/Anima/%"
                else "${MediaStore.MediaColumns.RELATIVE_PATH} LIKE ?" to Environment.DIRECTORY_PICTURES + "/Anima%"
            ctx.contentResolver.query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, arrayOf(MediaStore.Images.Media._ID),
                where, arrayOf(arg), null)?.use { c ->
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

    /** Several pictures at once, with the list rewritten once rather than once per picture. */
    fun deleteMany(ctx: Context, es: Collection<Entry>) {
        es.forEach { e -> runCatching { ctx.contentResolver.delete(e.uri, null, null) } }
        val gone = es.map { it.uri }.toSet()
        write(ctx, read(ctx).filter { it.uri !in gone })
    }

    // Small previews already made, so reopening the gallery shows them at once (about 48 MB at most).
    private val thumbs = object : android.util.LruCache<String, Bitmap>(48 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap) = value.allocationByteCount
    }

    fun cachedThumbnail(e: Entry, px: Int): Bitmap? = thumbs.get("${e.uri}@$px")

    fun thumbnail(ctx: Context, e: Entry, px: Int): Bitmap? {
        cachedThumbnail(e, px)?.let { return it }
        return runCatching { if (legacy) legacyThumbnail(ctx, e, px) else ctx.contentResolver.loadThumbnail(e.uri, Size(px, px), null) }
            .getOrNull()?.also { thumbs.put("${e.uri}@$px", it) }
    }

    /** Android 9: a preview read at a fraction of the picture's size (a clip's first frame, shrunk). */
    private fun legacyThumbnail(ctx: Context, e: Entry, px: Int): Bitmap? {
        if (isVideo(ctx, e)) return android.media.MediaMetadataRetriever().run {
            try { setDataSource(ctx, e.uri); getFrameAtTime(0) } finally { release() }
        }?.let { b -> val k = px.toFloat() / maxOf(b.width, b.height); if (k >= 1f) b else Bitmap.createScaledBitmap(b, maxOf(1, (b.width * k).toInt()), maxOf(1, (b.height * k).toInt()), true) }
        val o = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
        ctx.contentResolver.openInputStream(e.uri)?.use { android.graphics.BitmapFactory.decodeStream(it, null, o) }
        var sample = 1
        while (maxOf(o.outWidth, o.outHeight) / (sample * 2) >= px) sample *= 2
        val o2 = android.graphics.BitmapFactory.Options().apply { inSampleSize = sample }
        return ctx.contentResolver.openInputStream(e.uri)?.use { android.graphics.BitmapFactory.decodeStream(it, null, o2) }
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
