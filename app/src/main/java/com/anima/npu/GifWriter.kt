package com.anima.npu

import java.io.OutputStream

/**
 * Writes a looping animated GIF from ARGB frames. One 256-colour table for the whole clip (picked by splitting the
 * clip's colours into boxes until there are 256), so colours do not flicker between frames; a fixed 4x4 pattern
 * softens banding without the crawling noise that error spreading gives in a moving picture.
 */
object GifWriter {
    /** [frames]: w*h ARGB each. [pingPong]: play forward then back, so the loop has no jump. [delayCs]: per frame, 1/100 s. */
    fun write(out: OutputStream, frames: List<IntArray>, w: Int, h: Int, delayCs: Int = 4, pingPong: Boolean = true) {
        val order = if (pingPong && frames.size > 2) frames.indices.toList() + (frames.size - 2 downTo 1) else frames.indices.toList()
        val palette = palette(frames)
        val lut = lookup(palette)
        val o = out.buffered(1 shl 16)
        o.write("GIF89a".toByteArray())
        short(o, w); short(o, h)
        o.write(0xF7); o.write(0); o.write(0)          // global table of 256, background 0, square pixels
        for (c in palette) { o.write(c shr 16 and 255); o.write(c shr 8 and 255); o.write(c and 255) }
        o.write(byteArrayOf(0x21, 0xFF.toByte(), 11)); o.write("NETSCAPE2.0".toByteArray())
        o.write(byteArrayOf(3, 1, 0, 0, 0))            // loop for ever
        val idx = ByteArray(w * h)
        for (f in order) {
            val px = frames[f]
            for (y in 0 until h) for (x in 0 until w) {
                val c = px[y * w + x]
                val d = BAYER[(y and 3) * 4 + (x and 3)] - 8   // -8..7: a third of a 5-bit step either way
                val r = ((c shr 16 and 255) + d).coerceIn(0, 255)
                val g = ((c shr 8 and 255) + d).coerceIn(0, 255)
                val b = ((c and 255) + d).coerceIn(0, 255)
                idx[y * w + x] = lut[(r shr 3 shl 10) or (g shr 3 shl 5) or (b shr 3)]
            }
            o.write(byteArrayOf(0x21, 0xF9.toByte(), 4, 0x04)); short(o, delayCs); o.write(0); o.write(0)
            o.write(0x2C); short(o, 0); short(o, 0); short(o, w); short(o, h); o.write(0)
            o.write(8)
            lzw(o, idx)
            o.write(0)
        }
        o.write(0x3B)
        o.flush()
    }

    private val BAYER = intArrayOf(0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5)

    private fun short(o: OutputStream, v: Int) { o.write(v and 255); o.write(v shr 8 and 255) }

    /** 256 colours (RGB ints) by splitting the clip's 15-bit colour counts along their widest side at the middle count. */
    private fun palette(frames: List<IntArray>): IntArray {
        val hist = IntArray(32768)
        for (px in frames) {
            var i = 0
            while (i < px.size) { val c = px[i]; hist[(c shr 19 and 31 shl 10) or (c shr 11 and 31 shl 5) or (c shr 3 and 31)]++; i += 3 }
        }
        val cols = (0 until 32768).filter { hist[it] > 0 }.toIntArray()
        if (cols.size <= 256) return IntArray(256) { if (it < cols.size) expand(cols[it]) else 0 }
        // boxes: [from, to) ranges of cols, sorted along the box's widest channel when split
        val boxes = ArrayList<IntArray>().apply { add(intArrayOf(0, cols.size)) }
        fun ch(c: Int, k: Int) = c shr (10 - 5 * k) and 31
        fun score(b: IntArray): Long {
            var n = 0L; var span = 0
            for (k in 0..2) {
                var lo = 31; var hi = 0
                for (i in b[0] until b[1]) { val v = ch(cols[i], k); if (v < lo) lo = v; if (v > hi) hi = v }
                span = maxOf(span, hi - lo)
            }
            for (i in b[0] until b[1]) n += hist[cols[i]]
            return if (b[1] - b[0] < 2) -1 else n * span
        }
        val scores = ArrayList<Long>().apply { add(score(boxes[0])) }
        while (boxes.size < 256) {
            val bi = scores.indices.maxByOrNull { scores[it] }!!
            if (scores[bi] <= 0) break
            val b = boxes[bi]
            var wide = 0; var best = -1
            for (k in 0..2) {
                var lo = 31; var hi = 0
                for (i in b[0] until b[1]) { val v = ch(cols[i], k); if (v < lo) lo = v; if (v > hi) hi = v }
                if (hi - lo > best) { best = hi - lo; wide = k }
            }
            val part = cols.copyOfRange(b[0], b[1]).sortedBy { ch(it, wide) }
            for (i in part.indices) cols[b[0] + i] = part[i]
            var total = 0L
            for (i in b[0] until b[1]) total += hist[cols[i]]
            var acc = 0L; var cut = b[0] + 1
            for (i in b[0] until b[1] - 1) { acc += hist[cols[i]]; if (acc * 2 >= total) { cut = i + 1; break } }
            val a = intArrayOf(b[0], cut); val c = intArrayOf(cut, b[1])
            boxes[bi] = a; scores[bi] = score(a)
            boxes.add(c); scores.add(score(c))
        }
        return IntArray(256) { i ->
            if (i >= boxes.size) 0 else {
                val b = boxes[i]
                var n = 0L; var r = 0L; var g = 0L; var bl = 0L
                for (j in b[0] until b[1]) {
                    val c = cols[j]; val k = hist[c].toLong(); val e = expand(c)
                    n += k; r += k * (e shr 16 and 255); g += k * (e shr 8 and 255); bl += k * (e and 255)
                }
                ((r / n).toInt() shl 16) or ((g / n).toInt() shl 8) or (bl / n).toInt()
            }
        }
    }

    private fun expand(c15: Int): Int {
        val r = (c15 shr 10 and 31) shl 3 or 4; val g = (c15 shr 5 and 31) shl 3 or 4; val b = (c15 and 31) shl 3 or 4
        return (r shl 16) or (g shl 8) or b
    }

    /** For every 15-bit colour, the nearest palette entry. */
    private fun lookup(pal: IntArray): ByteArray = ByteArray(32768) { c ->
        val e = expand(c)
        val r = e shr 16 and 255; val g = e shr 8 and 255; val b = e and 255
        var best = 0; var bd = Int.MAX_VALUE
        for (i in pal.indices) {
            val p = pal[i]
            val dr = r - (p shr 16 and 255); val dg = g - (p shr 8 and 255); val db = b - (p and 255)
            val d = 2 * dr * dr + 4 * dg * dg + 3 * db * db
            if (d < bd) { bd = d; best = i }
        }
        best.toByte()
    }

    private val stamp = IntArray(1 shl 20)
    private val codeOf = IntArray(1 shl 20)
    private var gen = 0

    /** GIF's word-table compression of one frame's colour numbers (8-bit start), in blocks of up to 255 bytes. */
    @Synchronized
    private fun lzw(o: OutputStream, idx: ByteArray) {
        val block = ByteArray(255); var fill = 0
        var acc = 0; var bits = 0
        fun emit(code: Int, size: Int) {
            acc = acc or (code shl bits); bits += size
            while (bits >= 8) {
                block[fill++] = acc.toByte(); acc = acc ushr 8; bits -= 8
                if (fill == 255) { o.write(255); o.write(block, 0, 255); fill = 0 }
            }
        }
        val clear = 256; val eoi = 257
        var size = 9; var next = eoi + 1
        gen++
        emit(clear, size)
        var cur = idx[0].toInt() and 255
        for (i in 1 until idx.size) {
            val k = idx[i].toInt() and 255
            val key = (cur shl 8) or k
            if (stamp[key] == gen) { cur = codeOf[key]; continue }
            emit(cur, size)
            if (next == 4096) {
                emit(clear, size); next = eoi + 1; size = 9; gen++
            } else {
                if (next >= (1 shl size)) size++
                stamp[key] = gen; codeOf[key] = next++
            }
            cur = k
        }
        emit(cur, size)
        emit(eoi, size)
        if (bits > 0) { block[fill++] = acc.toByte(); if (fill == 255) { o.write(255); o.write(block, 0, 255); fill = 0 } }
        if (fill > 0) { o.write(fill); o.write(block, 0, fill) }
    }
}
