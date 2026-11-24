package com.anima.npu

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ThemeTest {
    @Test fun blackIsPureBlack() = assertEquals(Color(0xFF000000), paperColors(true).background)

    @Test fun lightBackground() = assertEquals(Color(0xFFF4F5F7), paperColors(false).background)

    @Test fun accentSameInBoth() {
        assertEquals(Color(0xFF3B5BDB), paperColors(true).accent)
        assertEquals(Color(0xFF3B5BDB), paperColors(false).accent)
    }

    @Test fun blackCardsHaveBorderNotShadow() {
        val p = paperColors(true)
        assertEquals(Color(0xFF1F2125), p.cardLine)
        assertFalse(p.cardShadow)
        assertTrue(paperColors(false).cardShadow)
    }

    @Test fun readableText() {
        for (black in listOf(true, false)) {
            val p = paperColors(black)
            for ((name, c) in listOf("text" to p.text, "muted" to p.muted, "link" to p.link)) {
                assertTrue("$name on background, black=$black", contrast(c, p.background) >= 4.5)
                assertTrue("$name on card, black=$black", contrast(c, p.card) >= 4.5)
            }
        }
    }

    @Test fun schemeUsesPaperBackground() = assertEquals(Color(0xFF000000), paperScheme(paperColors(true), true).background)

    private fun contrast(a: Color, b: Color): Double {
        val la = a.luminance() + 0.05
        val lb = b.luminance() + 0.05
        return maxOf(la, lb).toDouble() / minOf(la, lb)
    }
}
