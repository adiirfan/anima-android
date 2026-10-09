package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UpdateInfoTest {
    @Test fun parses() {
        val u = UpdateInfo.parse("""{"build": 1791500000, "size": 57000000, "sha256": "ab12", "notes": "Queue"}""")
        assertEquals(1791500000L, u.build); assertEquals(57000000L, u.size); assertEquals("ab12", u.sha256); assertEquals("Queue", u.notes)
    }
    @Test fun notesOptional() { assertEquals("", UpdateInfo.parse("""{"build": 5, "size": 1, "sha256": "x"}""").notes) }
    @Test fun newer() {
        val u = UpdateInfo(100, 1, "x", "")
        assertTrue(u.isNewerThan(99)); assertFalse(u.isNewerThan(100)); assertFalse(u.isNewerThan(101))
    }
    @Test fun unstampedBuildNeverOffered() { assertFalse(UpdateInfo(100, 1, "x", "").isNewerThan(0)) }
}
