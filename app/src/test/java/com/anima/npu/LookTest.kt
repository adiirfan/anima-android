package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LookTest {
    @Test fun parsesStoredValues() {
        assertEquals(ThemeMode.LIGHT, Look.parse("light"))
        assertEquals(ThemeMode.BLACK, Look.parse("black"))
        assertEquals(ThemeMode.PHONE, Look.parse("phone"))
    }

    @Test fun unknownOrMissingMeansPhone() {
        assertEquals(ThemeMode.PHONE, Look.parse(null))
        assertEquals(ThemeMode.PHONE, Look.parse("dark"))
    }

    @Test fun keyRoundTrips() {
        ThemeMode.values().forEach { assertEquals(it, Look.parse(Look.key(it))) }
    }

    @Test fun phoneFollowsSystem() {
        assertTrue(Look.isBlack(ThemeMode.PHONE, systemDark = true))
        assertFalse(Look.isBlack(ThemeMode.PHONE, systemDark = false))
        assertTrue(Look.isBlack(ThemeMode.BLACK, systemDark = false))
        assertFalse(Look.isBlack(ThemeMode.LIGHT, systemDark = true))
    }
}
