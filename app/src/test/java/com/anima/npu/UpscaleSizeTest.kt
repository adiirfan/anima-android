package com.anima.npu

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UpscaleSizeTest {
    @Test fun drawnPicturesCanBeUpscaled() {
        assertTrue(Upscaler.fits(768, 768))
        assertTrue(Upscaler.fits(576, 1024))
        assertTrue(Upscaler.fits(1024, 576))
    }

    // with "Upscale every new picture" on, a finished picture is already 4x: 4x again (12288 x 12288) ran the app
    // out of memory and crashed it (release check, 2026-11-24)
    @Test fun upscaledPicturesCannotBeUpscaledAgain() {
        assertFalse(Upscaler.fits(3072, 3072))
        assertFalse(Upscaler.fits(2304, 4096))
        assertFalse(Upscaler.fits(1025, 1025))
    }
}
