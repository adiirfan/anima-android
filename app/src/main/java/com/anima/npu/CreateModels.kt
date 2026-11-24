package com.anima.npu

/** One row of the Create page's model menu: the model's id, Qwen's drawing files ("qwen" or "noctq"), and the name shown. */
data class ModelChoice(val id: String, val drawing: String?, val name: String)

/**
 * Every model the Create page offers, in menu order. A model with nothing for this phone is still listed (as
 * unavailable). Noct Q is Qwen with other drawing files copied over the cable, so it is listed only when present.
 * A downloadable model missing here cannot be used at all: CreateModelsTest checks every one is listed.
 */
fun createChoices(noctq: Boolean): List<ModelChoice> = listOfNotNull(
    ModelChoice("anima", null, "Anima"),
    ModelChoice("klein", null, "FLUX.2 klein"),
    ModelChoice("krea", null, "Krea 2 Turbo"),
    ModelChoice("qwen", "qwen", "Qwen-Image 2.1"),
    ModelChoice("qwen", "noctq", "Noct Q").takeIf { noctq },
    ModelChoice("fibo", null, "Fibo 2.0 (edits)"),
    ModelChoice("moss", null, "MOSS-Image"),
    ModelChoice("wan", null, "Wan 2.2 (moving picture)"),
)

/** Models that only change a picture: Draw stays off until one is chosen. */
val NEEDS_PICTURE = setOf("fibo", "wan")

/** Models that draw from words only: no picture-to-edit row. */
val TEXT_ONLY = setOf("anima", "krea", "moss")
