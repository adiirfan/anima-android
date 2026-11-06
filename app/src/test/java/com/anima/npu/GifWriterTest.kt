package com.anima.npu

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** Writes a small moving gradient as a GIF to build/gif_test.gif; a Python check decodes it and compares. */
class GifWriterTest {
    @Test fun writesGif() {
        val w = 160; val h = 96
        val frames = List(6) { f ->
            IntArray(w * h) { i ->
                val x = i % w; val y = i / w
                val r = (x * 255 / w + f * 20).coerceAtMost(255); val g = y * 255 / h; val b = ((x + y + f * 10) * 2) and 255
                (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val out = File("build/gif_test.gif")
        out.outputStream().use { GifWriter.write(it, frames, w, h) }
        File("build/gif_test.raw").outputStream().use { o ->
            frames.forEach { px -> px.forEach { c -> o.write(c shr 16 and 255); o.write(c shr 8 and 255); o.write(c and 255) } }
        }
        assertTrue(out.length() > 1000)
    }
}
