package com.anima.npu

import android.os.Build

/**
 * Which neural unit this phone has. The compiled model files only run on the chip they were built for:
 * Snapdragon 8 Elite (Hexagon v79) uses the original repos, 8 Elite Gen 5 (Hexagon v81) the "-gen5" ones.
 * Any other chip has no build, so the app says the device is not supported instead of failing at load.
 */
object Chip {
    enum class Kind { ELITE, ELITE_GEN5, UNSUPPORTED }

    /** The chip's model name as Android reports it, e.g. "SM8850" (empty on Android 11 and older). */
    val soc: String = if (Build.VERSION.SDK_INT >= 31) (Build.SOC_MODEL ?: "").uppercase() else ""

    private val detected: Kind = when {
        soc.startsWith("SM8750") -> Kind.ELITE
        soc.startsWith("SM8850") -> Kind.ELITE_GEN5
        else -> Kind.UNSUPPORTED
    }

    /** Test builds only: files/test-unsupported (made over the cable with run-as) makes the app act as on an unsupported phone. */
    var testOverride: Kind? = null

    val kind: Kind get() = testOverride ?: detected

    val supported get() = kind != Kind.UNSUPPORTED

    /** A name people recognise, for messages. */
    val label: String
        get() = when (kind) {
            Kind.ELITE -> "Snapdragon 8 Elite"
            Kind.ELITE_GEN5 -> "Snapdragon 8 Elite Gen 5"
            Kind.UNSUPPORTED -> listOfNotNull(
                if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER?.takeIf { it.isNotBlank() && it != Build.UNKNOWN } else null,
                soc.ifEmpty { null },
            ).joinToString(" ").ifEmpty { "this phone's chip" }
        }

    const val SUPPORTED_TEXT = "Anima runs on phones with a Snapdragon 8 Elite or Snapdragon 8 Elite Gen 5."
}
