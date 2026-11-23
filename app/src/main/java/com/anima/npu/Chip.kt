package com.anima.npu

import android.os.Build

/**
 * Which neural unit this phone has. The compiled model files only run on the chip they were built for:
 * Snapdragon 8 Elite (Hexagon v79) uses the original repos, 8 Elite Gen 5 (Hexagon v81) the "-gen5" ones. The 8 Gen 3
 * (v75) and 8 Gen 2 (v73) have Anima only ("-8gen3" / "-8gen2" repos, built without a phone to test on, 2026-11-23);
 * everything else on them runs on the graphics chip. Any other chip has no build, so the app says the device is not
 * supported instead of failing at load.
 */
object Chip {
    enum class Kind { ELITE, ELITE_GEN5, GEN3, GEN2, UNSUPPORTED }

    /** The chip's model name as Android reports it, e.g. "SM8850" (empty on Android 11 and older). */
    val soc: String = if (Build.VERSION.SDK_INT >= 31) (Build.SOC_MODEL ?: "").uppercase() else ""

    private val detected: Kind = when {
        soc.startsWith("SM8750") -> Kind.ELITE
        soc.startsWith("SM8850") -> Kind.ELITE_GEN5
        soc.startsWith("SM8650") -> Kind.GEN3
        soc.startsWith("SM8550") -> Kind.GEN2
        else -> Kind.UNSUPPORTED
    }

    /** Test builds only: files/test-unsupported (made over the cable with run-as) makes the app act as on an unsupported phone. */
    var testOverride: Kind? = null

    val kind: Kind get() = testOverride ?: detected

    /** Every AI-chip feature (the chat and MOSS's reader on the chip, the upscaler, all the models' AI-chip builds). */
    val supported get() = kind == Kind.ELITE || kind == Kind.ELITE_GEN5

    /** The 8 Gen 3 / 8 Gen 2: only Anima has an AI-chip build; the rest runs on the graphics chip. */
    val animaOnly get() = kind == Kind.GEN3 || kind == Kind.GEN2

    /** A name people recognise, for messages. */
    val label: String
        get() = when (kind) {
            Kind.ELITE -> "Snapdragon 8 Elite"
            Kind.ELITE_GEN5 -> "Snapdragon 8 Elite Gen 5"
            Kind.GEN3 -> "Snapdragon 8 Gen 3"
            Kind.GEN2 -> "Snapdragon 8 Gen 2"
            Kind.UNSUPPORTED -> listOfNotNull(
                if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER?.takeIf { it.isNotBlank() && it != Build.UNKNOWN } else null,
                soc.ifEmpty { null },
            ).joinToString(" ").ifEmpty { "this phone's chip" }
        }

    const val SUPPORTED_TEXT = "Anima runs on phones with a Snapdragon 8 Elite, 8 Elite Gen 5, 8 Gen 3 or 8 Gen 2."
}
