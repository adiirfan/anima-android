package com.anima.npu

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.withContext

/**
 * The phone's 4x upscaler (XLSR, in Anima's download: models/anima/xlsr.bin), for any picture: it works through the
 * picture in 512-pixel tiles. Used for "Upscale every new picture" (Settings), the Gallery's Upscale button and
 * "upscale" queue jobs.
 */
object Upscaler {
    /** Upscales from the Gallery run here, so leaving the screen does not drop the result. */
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private var state: MutableState<Boolean>? = null
    private fun st(ctx: Context) = state ?: mutableStateOf(
        ctx.getSharedPreferences("anima", Context.MODE_PRIVATE).getBoolean("autoUpscale", false)).also { state = it }

    /** Every new picture is upscaled 4x before it is saved. */
    fun auto(ctx: Context) = st(ctx).value
    fun setAuto(ctx: Context, on: Boolean) {
        st(ctx).value = on
        ctx.getSharedPreferences("anima", Context.MODE_PRIVATE).edit().putBoolean("autoUpscale", on).apply()
    }

    fun available(ctx: Context) = ModelStore.ANIMA.installed(ctx) != null

    /** The picture 4x bigger. Throws with the reason when it cannot. */
    suspend fun upscale(ctx: Context, bmp: Bitmap): Bitmap = withContext(Dispatchers.Default) {
        if (!available(ctx)) error("The upscaler comes with Anima: download Anima first")
        Engines.ensureAnima()?.let { error("Could not start the upscaler: $it") }
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }
        val up = Engine.upscale(px, w, h) ?: error(Engine.lastError())
        Bitmap.createBitmap(up, 4 * w, 4 * h, Bitmap.Config.ARGB_8888)
    }
}
