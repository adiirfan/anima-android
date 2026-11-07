package com.anima.npu

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.FileDescriptor

/**
 * A clip as an H.264 MP4 (the phone's own video encoder). Replaces the GIF: a GIF holds 256 colours, which turned
 * faces into flat patches. The frames play forward then back ([pingPong]) so the loop has no jump, [loops] times over,
 * so players that do not loop still show a few seconds.
 */
object VideoWriter {
    fun write(fd: FileDescriptor, frames: List<IntArray>, w: Int, h: Int, fps: Int = 24, loops: Int = 3, pingPong: Boolean = true) {
        val order = if (pingPong && frames.size > 2) frames.indices.toList() + (frames.size - 2 downTo 1) else frames.indices.toList()
        val seq = List(loops) { order }.flatten()
        val fmt = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, 12_000_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        codec.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
        val muxer = MediaMuxer(fd, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var track = -1
        val info = MediaCodec.BufferInfo()
        var next = 0
        var inputDone = false
        try {
            while (true) {
                if (!inputDone) {
                    val i = codec.dequeueInputBuffer(10_000)
                    if (i >= 0) {
                        val pts = next * 1_000_000L / fps
                        if (next == seq.size) {
                            codec.queueInputBuffer(i, 0, 0, pts, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            val img = codec.getInputImage(i) ?: error("the video encoder gave no input picture")
                            fill(img, frames[seq[next]], w, h)
                            codec.queueInputBuffer(i, 0, w * h * 3 / 2, pts, 0)
                            next++
                        }
                    }
                }
                val o = codec.dequeueOutputBuffer(info, 10_000)
                when {
                    o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { track = muxer.addTrack(codec.outputFormat); muxer.start() }
                    o >= 0 -> {
                        val buf = codec.getOutputBuffer(o)!!
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                        if (info.size > 0 && track >= 0) {
                            buf.position(info.offset); buf.limit(info.offset + info.size)
                            muxer.writeSampleData(track, buf, info)
                        }
                        codec.releaseOutputBuffer(o, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                    }
                }
            }
        } finally {
            runCatching { codec.stop() }; codec.release()
            runCatching { if (track >= 0) muxer.stop() }; muxer.release()
        }
    }

    /** ARGB -> the encoder's YUV 4:2:0 picture (BT.601, limited range), whatever its plane layout. */
    private fun fill(img: android.media.Image, argb: IntArray, w: Int, h: Int) {
        val (yp, up, vp) = img.planes
        val y = yp.buffer; val u = up.buffer; val v = vp.buffer
        for (r in 0 until h) {
            var at = r * yp.rowStride
            for (c in 0 until w) {
                val p = argb[r * w + c]
                val R = (p shr 16) and 255; val G = (p shr 8) and 255; val B = p and 255
                y.put(at, (((66 * R + 129 * G + 25 * B + 128) shr 8) + 16).toByte())
                at += yp.pixelStride
            }
        }
        for (r in 0 until h / 2) for (c in 0 until w / 2) {
            var R = 0; var G = 0; var B = 0
            for (dy in 0..1) for (dx in 0..1) {
                val p = argb[(2 * r + dy) * w + 2 * c + dx]
                R += (p shr 16) and 255; G += (p shr 8) and 255; B += p and 255
            }
            R /= 4; G /= 4; B /= 4
            u.put(r * up.rowStride + c * up.pixelStride, (((-38 * R - 74 * G + 112 * B + 128) shr 8) + 128).toByte())
            v.put(r * vp.rowStride + c * vp.pixelStride, (((112 * R - 94 * G - 18 * B + 128) shr 8) + 128).toByte())
        }
    }
}
