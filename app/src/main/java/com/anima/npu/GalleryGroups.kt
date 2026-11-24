package com.anima.npu

import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * The gallery's filter pills (2026-11-24). A clip (GIF or video) is only ever a Clip. An edit is a picture made from
 * another picture by an editing model, or an upscale; pictures saved before the gallery remembered this count as
 * plain Pictures.
 */
enum class GalleryFilter(val label: String) {
    ALL("All"), PICTURES("Pictures"), CLIPS("Clips"), EDITS("Edits");

    fun matches(kind: String, isClip: Boolean): Boolean = when (this) {
        ALL -> true
        CLIPS -> isClip
        EDITS -> !isClip && kind == "edit"
        PICTURES -> !isClip && kind != "edit"
    }
}

private val SAME_YEAR = DateTimeFormatter.ofPattern("d MMM", Locale.ENGLISH)
private val OTHER_YEAR = DateTimeFormatter.ofPattern("d MMM yyyy", Locale.ENGLISH)

/** "Today", "Yesterday", "21 Nov", or "3 Dec 2025" in another year; "Earlier" for a picture saved without a date. */
fun dayLabel(savedAt: Long, now: Long, zone: ZoneId): String {
    if (savedAt <= 0) return "Earlier"
    val day = Instant.ofEpochMilli(savedAt).atZone(zone).toLocalDate()
    val today = Instant.ofEpochMilli(now).atZone(zone).toLocalDate()
    return when {
        day == today -> "Today"
        day == today.minusDays(1) -> "Yesterday"
        day.year == today.year -> day.format(SAME_YEAR)
        else -> day.format(OTHER_YEAR)
    }
}

/** Runs of items saved on the same day, in the order given (the gallery is newest first), each under its day label. */
fun <T> groupByDay(items: List<T>, savedAt: (T) -> Long, now: Long, zone: ZoneId): List<Pair<String, List<T>>> {
    val out = mutableListOf<Pair<String, MutableList<T>>>()
    var lastDay: LocalDate? = null
    for (item in items) {
        val day = if (savedAt(item) <= 0) LocalDate.MIN else Instant.ofEpochMilli(savedAt(item)).atZone(zone).toLocalDate()
        if (day != lastDay) { out += dayLabel(savedAt(item), now, zone) to mutableListOf(); lastDay = day }
        out.last().second += item
    }
    return out
}
