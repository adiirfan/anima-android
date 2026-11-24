package com.anima.npu

import android.content.Context
import java.io.File

/**
 * How the app's programs (llama-server, the prompt readers, the graphics engine) find their libraries.
 *
 * The bundled llama.cpp set is built for 2017-era processors (Android 9 phones such as the Snapdragon 845), so it
 * runs on every phone. Processors with the newer maths instructions (8-bit matrix and dot product, Snapdragon 8 Gen 1
 * on) also get the fast build of its maths part, libggml-cpu-fast.so: it is loaded first, and since it carries the
 * name libggml-cpu.so inside, the other libraries use it in place of the slow one.
 */
object NativeLibs {
    @Volatile private var fast: Boolean? = null

    /** The processor has 8-bit matrix (i8mm) and dot-product (asimddp) instructions. */
    fun fastCpu(): Boolean = fast ?: runCatching {
        File("/proc/cpuinfo").useLines { l -> l.firstOrNull { it.startsWith("Features") } }
            ?.substringAfter(':')?.trim()?.split(Regex("\\s+"))?.let { "i8mm" in it && "asimddp" in it } ?: false
    }.getOrDefault(false).also { fast = it }

    /** Sets LD_LIBRARY_PATH (and LD_PRELOAD on fast processors) for a program the app starts. */
    fun apply(ctx: Context, env: MutableMap<String, String>) {
        val libs = ctx.applicationInfo.nativeLibraryDir
        env["LD_LIBRARY_PATH"] = "$libs:/system/lib64:/vendor/lib64:/system/vendor/lib64"
        val f = File(libs, "libggml-cpu-fast.so")
        if (fastCpu() && f.isFile) env["LD_PRELOAD"] = f.absolutePath
    }
}
