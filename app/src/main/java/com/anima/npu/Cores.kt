package com.anima.npu

import java.io.File

/**
 * How many cores to give chat on the processor: the fast ones only. Phones pair fast cores with slow ones, and
 * letting the slow ones join made the XZ3 crawl (2026-11-24). [maxFreqKhz] is each core's top
 * speed; zeros (unreadable) are ignored. All cores alike means all of them; nothing readable means 4.
 */
internal fun fastCores(maxFreqKhz: List<Long>): Int {
    val known = maxFreqKhz.filter { it > 0 }
    if (known.isEmpty()) return 4
    val slowest = known.min()
    val fast = known.count { it > slowest }
    return if (fast == 0) known.size else fast
}

/** The top speed of each of this phone's cores, as Android lists them. */
internal fun coreMaxFreqs(): List<Long> = (0 until 32).map { i ->
    File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq")
}.takeWhile { it.parentFile?.parentFile?.isDirectory == true }.map { f ->
    runCatching { f.readText().trim().toLong() }.getOrDefault(0L)
}
