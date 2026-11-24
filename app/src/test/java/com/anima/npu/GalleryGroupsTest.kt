package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.ZoneId
import java.time.ZonedDateTime

class GalleryGroupsTest {
    private val zone = ZoneId.of("Asia/Jakarta")
    private val now = ZonedDateTime.of(2026, 11, 24, 10, 0, 0, 0, zone).toInstant().toEpochMilli()
    private fun at(d: Int, h: Int, y: Int = 2026, m: Int = 11) = ZonedDateTime.of(y, m, d, h, 0, 0, 0, zone).toInstant().toEpochMilli()

    @Test fun filters() {
        assertTrue(GalleryFilter.ALL.matches("edit", false))
        assertTrue(GalleryFilter.PICTURES.matches("picture", false))
        assertFalse(GalleryFilter.PICTURES.matches("edit", false))
        assertFalse(GalleryFilter.PICTURES.matches("picture", true))
        assertTrue(GalleryFilter.CLIPS.matches("picture", true))
        assertTrue(GalleryFilter.EDITS.matches("edit", false))
        assertFalse(GalleryFilter.EDITS.matches("picture", false))
    }

    @Test fun clipIsNeverAnEditOrPicture() {
        assertFalse(GalleryFilter.EDITS.matches("edit", true))
        assertTrue(GalleryFilter.CLIPS.matches("edit", true))
    }

    @Test fun labels() {
        assertEquals("Today", dayLabel(at(24, 0), now, zone))
        assertEquals("Yesterday", dayLabel(at(23, 23), now, zone))
        assertEquals("21 Nov", dayLabel(at(21, 12), now, zone))
        assertEquals("3 Dec 2025", dayLabel(at(3, 12, 2025, 12), now, zone))
    }

    @Test fun groupsKeepOrder() {
        val items = listOf(at(24, 9), at(24, 1), at(23, 5), at(21, 5))
        val g = groupByDay(items, { it }, now, zone)
        assertEquals(listOf("Today", "Yesterday", "21 Nov"), g.map { it.first })
        assertEquals(listOf(at(24, 9), at(24, 1)), g[0].second)
    }

    @Test fun unknownDateIsEarlier() = assertEquals("Earlier", dayLabel(0, now, zone))

    @Test fun emptyGivesNoGroups() = assertEquals(0, groupByDay(emptyList<Long>(), { it }, now, zone).size)
}
