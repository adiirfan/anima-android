package com.anima.npu

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.CpuHeadroomParams
import android.os.GpuHeadroomParams
import android.os.health.SystemHealthManager
import android.system.Os
import android.system.OsConstants
import androidx.compose.foundation.clickable
import androidx.compose.ui.Alignment
import androidx.compose.material3.TextButton
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File

/**
 * One reading of what the phone is doing, as far as an app may see it. The whole phone's processor and graphics use
 * come from /proc/stat and the kgsl busy file when the app may read them, else from Android 16's headroom calls
 * (busy = 100 - headroom), else they are missing (null). On the iQOO 13 all of these are refused (the files even to
 * adb; "GPU Headroom Supported: false"; no power meters), so there the bar shows the app's own processor share, the
 * neural unit time our engine spent running graphs, and each part's temperature (thermal zones cpu-*, gpuss-*,
 * nsp* are readable), which shows the graphics chip working when its load cannot be read.
 */
private class Usage(
    val cpuPct: Int?, val appCpuPct: Int, val cpuGhz: Double, val gpuPct: Int?, val npuPct: Int, val appMb: Long,
    val freeMb: Long, val totalMb: Long, val tempC: Double, val cpuC: Double?, val gpuC: Double?, val npuC: Double?,
    val watts: Double?, val charging: Boolean, val diskReadMBs: Double, val diskWriteMBs: Double, val diskWhole: Boolean,
)

private class Sampler(private val ctx: Context) {
    private val cores = Runtime.getRuntime().availableProcessors()
    private val tick = Os.sysconf(OsConstants._SC_CLK_TCK).toDouble()
    private var lastWall = System.nanoTime()
    private var lastCpu = cpuTicks()
    private var lastNpu = Engine.npuBusyNs()
    private var lastStat = procStat()
    private val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private var lastIo = diskBytes()

    /**
     * (read, written) bytes, and whether they are the whole phone's: /proc/diskstats (sectors of 512 bytes, whole
     * disks only) when the app may read it, else this app's own storage traffic from /proc/self/io (which includes
     * the model files it maps and reads; the iQOO 13 allows only this).
     */
    private fun diskBytes(): Triple<Long, Long, Boolean> {
        runCatching {
            var r = 0L; var w = 0L
            File("/proc/diskstats").readLines().forEach { l ->
                val f = l.trim().split(Regex("\\s+"))
                if (f.size > 9 && Regex("^(sd[a-z]+|mmcblk\\d+|nvme\\d+n\\d+)$").matches(f[2])) { r += f[5].toLong() * 512; w += f[9].toLong() * 512 }
            }
            if (r > 0) return Triple(r, w, true)
        }
        return runCatching {
            val m = File("/proc/self/io").readLines().associate { it.substringBefore(":") to it.substringAfter(":").trim().toLong() }
            Triple(m["read_bytes"] ?: 0L, m["write_bytes"] ?: 0L, false)
        }.getOrDefault(Triple(0L, 0L, false))
    }

    /** Battery power in watts (current x voltage) and whether it is charging; null when the phone does not say. */
    private fun watts(volts: Double): Pair<Double?, Boolean> {
        val ua = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val charging = bm.isCharging
        if (ua == Long.MIN_VALUE || ua == 0L && volts <= 0) return null to charging
        // some phones report mA instead of µA
        val amps = if (kotlin.math.abs(ua) > 20_000) ua / 1e6 else ua / 1e3
        return kotlin.math.abs(amps * volts) to charging
    }
    private val health = if (Build.VERSION.SDK_INT >= 36) ctx.getSystemService(SystemHealthManager::class.java) else null
    private var cpuHeadroom = true                // false once the phone says the call is not supported
    private var gpuHeadroom = true

    // temperature sensors by part: the readable thermal zones whose type starts with these
    private val zones: Map<String, List<File>> = run {
        val all = File("/sys/class/thermal").listFiles { f -> f.name.startsWith("thermal_zone") }.orEmpty().mapNotNull { z ->
            val type = runCatching { File(z, "type").readText().trim() }.getOrNull() ?: return@mapNotNull null
            if (runCatching { File(z, "temp").readText() }.isSuccess) type to File(z, "temp") else null
        }
        mapOf("cpu" to all.filter { it.first.startsWith("cpu-") }.map { it.second },
            "gpu" to all.filter { it.first.startsWith("gpuss-") }.map { it.second },
            "npu" to all.filter { it.first.startsWith("nsp") }.map { it.second })
    }

    private fun hottest(part: String): Double? = zones[part].orEmpty().mapNotNull { f ->
        runCatching { f.readText().trim().toLong() / 1000.0 }.getOrNull()
    }.maxOrNull()

    /** (busy, total) jiffies of all cores from /proc/stat, or null when the app may not read it */
    private fun procStat(): Pair<Long, Long>? = runCatching {
        val v = File("/proc/stat").readLines().first().split(Regex("\\s+")).drop(1).map { it.toLong() }
        val idle = v[3] + v[4]
        (v.sum() - idle) to v.sum()
    }.getOrNull()

    private fun phoneCpu(): Int? {
        val now = procStat()
        val last = lastStat
        lastStat = now
        if (now != null && last != null && now.second > last.second)
            return ((now.first - last.first) * 100 / (now.second - last.second)).toInt().coerceIn(0, 100)
        if (Build.VERSION.SDK_INT >= 36 && cpuHeadroom) runCatching {
            val h = health!!.getCpuHeadroom(CpuHeadroomParams.Builder().build())
            if (!h.isNaN()) return (100 - h).toInt().coerceIn(0, 100)
        }.onFailure { if (it is UnsupportedOperationException) cpuHeadroom = false }
        return null
    }

    private fun gpu(): Int? {
        runCatching { return File("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage").readText().filter { it.isDigit() }.toInt() }
        if (Build.VERSION.SDK_INT >= 36 && gpuHeadroom) runCatching {
            val h = health!!.getGpuHeadroom(GpuHeadroomParams.Builder().build())
            if (!h.isNaN()) return (100 - h).toInt().coerceIn(0, 100)
        }.onFailure { if (it is UnsupportedOperationException) gpuHeadroom = false }
        return null
    }

    private fun cpuTicks(): Long = runCatching {
        // after the name in brackets: field 14 utime, 15 stime (counted from the state field, the 3rd)
        val f = File("/proc/self/stat").readText().substringAfterLast(") ").split(" ")
        f[11].toLong() + f[12].toLong()
    }.getOrDefault(0L)

    private fun cpuGhz(): Double = (0 until cores).maxOfOrNull { c ->
        runCatching { File("/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq").readText().trim().toLong() }.getOrDefault(0L)
    }?.div(1e6) ?: 0.0

    private fun appMb(): Long = runCatching {
        File("/proc/self/status").readLines().first { it.startsWith("VmRSS:") }.filter { it.isDigit() }.toLong() / 1024
    }.getOrDefault(0L)

    fun read(): Usage {
        val wall = System.nanoTime()
        val cpu = cpuTicks()
        val npu = Engine.npuBusyNs()
        val dt = (wall - lastWall).coerceAtLeast(1)
        val cpuPct = ((cpu - lastCpu) / tick * 1e9 / dt / cores * 100).toInt().coerceIn(0, 100)
        val npuPct = ((npu - lastNpu) * 100.0 / dt).toInt().coerceIn(0, 100)
        lastWall = wall; lastCpu = cpu; lastNpu = npu
        val mi = ActivityManager.MemoryInfo().also { (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it) }
        val bat = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temp = bat?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)?.div(10.0) ?: 0.0
        val io = diskBytes()
        val last = lastIo
        lastIo = io
        val mbs = { now: Long, before: Long -> ((now - before).coerceAtLeast(0) / 1e6) / (dt / 1e9) }
        val (w, charging) = watts((bat?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0) / 1000.0)

        return Usage(phoneCpu(), cpuPct, cpuGhz(), gpu(), npuPct, appMb(), mi.availMem shr 20, mi.totalMem shr 20, temp,
            hottest("cpu"), hottest("gpu"), hottest("npu"), w, charging, mbs(io.first, last.first), mbs(io.second, last.second), io.third)
    }
}

/** A four-line usage bar, updated every second while it is on screen. */
@Composable
fun UsageMonitor(modifier: Modifier = Modifier) {
    val ctx = LocalContext.current
    val prefs = remember { ctx.getSharedPreferences("anima", android.content.Context.MODE_PRIVATE) }
    if (!StatsBar.shown(ctx)) return   // hidden: nothing on screen and nothing measured
    var open by remember { mutableStateOf(prefs.getBoolean("statsOpen", true)) }
    var u by remember { mutableStateOf<Usage?>(null) }
    LaunchedEffect(Unit) {
        val s = withContext(Dispatchers.IO) { Sampler(ctx) }
        while (true) {
            delay(1000)
            u = withContext(Dispatchers.IO) { s.read() }
        }
    }
    val style = MaterialTheme.typography.labelSmall
    // tap to shrink to one line or open again; Hide removes it (the Create screen has the switch to bring it back)
    Row(modifier.fillMaxWidth().padding(start = 16.dp), verticalAlignment = Alignment.Top) {
    Column(Modifier.weight(1f).clickable { open = !open; prefs.edit().putBoolean("statsOpen", open).apply() }
        .padding(vertical = 4.dp)) {
        val x = u
        if (x == null) {
            Text("Usage: measuring…", style = style)
        } else if (!open) {
            Text("NPU ${x.npuPct}% · " + (x.watts?.let { "%.1f W".format(it) } ?: "n/a") + " · battery %.0f° ▾".format(x.tempC),
                style = style, maxLines = 1)
        } else {
            val deg = { v: Double? -> v?.let { " %.0f°".format(it) } ?: "" }
            val cpu = x.cpuPct?.let { "$it% (app ${x.appCpuPct}%)" } ?: "app ${x.appCpuPct}%"
            val ghz = "%.1f GHz".format(x.cpuGhz)
            Text("CPU $cpu $ghz${deg(x.cpuC)}   GPU${x.gpuPct?.let { " $it%" } ?: ""}${deg(x.gpuC)}   NPU ${x.npuPct}%${deg(x.npuC)}",
                style = style, maxLines = 1)
            val power = x.watts?.let { "%.1f W%s".format(it, if (x.charging) " charging" else "") } ?: "n/a"
            Text("Power $power · battery %.0f°".format(x.tempC), style = style, maxLines = 1)
            Text("RAM app %.1f GB · free %.1f of %.0f GB".format(x.appMb / 1024.0, x.freeMb / 1024.0, x.totalMb / 1024.0),
                style = style, maxLines = 1)
            Text("Disk%s read %.0f MB/s · write %.0f MB/s ▴".format(if (x.diskWhole) "" else " (app)", x.diskReadMBs, x.diskWriteMBs),
                style = style, maxLines = 1)
        }
    }
    TextButton(onClick = { StatsBar.set(ctx, false) }) { Text("Hide", style = style) }
    }
}

/** Whether the usage bar is on screen at all (remembered; switched off by its Hide button, on from the Create screen). */
object StatsBar {
    private var state: androidx.compose.runtime.MutableState<Boolean>? = null
    private fun st(ctx: android.content.Context) = state ?: mutableStateOf(
        ctx.getSharedPreferences("anima", android.content.Context.MODE_PRIVATE).getBoolean("statsShown", true)).also { state = it }
    fun shown(ctx: android.content.Context) = st(ctx).value
    fun set(ctx: android.content.Context, on: Boolean) {
        st(ctx).value = on
        ctx.getSharedPreferences("anima", android.content.Context.MODE_PRIVATE).edit().putBoolean("statsShown", on).apply()
    }
}
