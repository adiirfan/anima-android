package com.anima.npu

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ShortcutInfo
import android.content.pm.ShortcutManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.drawable.Icon

/**
 * The app's icon. Android lets an app pick among icons built into it (one launcher entry per icon in the manifest,
 * only one switched on), which is how apps such as Reddit change theirs; it does not let an app make an arbitrary
 * picture its launcher icon. A picture from the gallery goes on the home screen as a shortcut instead.
 */
object AppIcon {
    /** (launcher entry, name, picture): the four built-in icons, drawn with Anima on the phone (2026-10-09). */
    val ALL = listOf(
        Triple("IconSilver", "Silver", R.drawable.icon_silver),
        Triple("IconPink", "Pink", R.drawable.icon_pink),
        Triple("IconBlack", "Black", R.drawable.icon_black),
        Triple("IconBlonde", "Blonde", R.drawable.icon_blonde),
    )

    private fun component(ctx: Context, alias: String) = ComponentName(ctx.packageName, "com.anima.npu.$alias")

    /** The launcher entry switched on now. */
    fun current(ctx: Context): String = ALL.map { it.first }.firstOrNull { a ->
        when (ctx.packageManager.getComponentEnabledSetting(component(ctx, a))) {
            PackageManager.COMPONENT_ENABLED_STATE_ENABLED -> true
            PackageManager.COMPONENT_ENABLED_STATE_DEFAULT -> a == ALL.first().first  // the manifest's default
            else -> false
        }
    } ?: ALL.first().first

    /**
     * Switches the launcher icon. The new entry goes on before the old one goes off, so there is never a moment with
     * no icon. Some launchers take a few seconds to show it, and drop a home-screen copy of the old icon.
     */
    fun set(ctx: Context, alias: String) {
        val pm = ctx.packageManager
        pm.setComponentEnabledSetting(component(ctx, alias), PackageManager.COMPONENT_ENABLED_STATE_ENABLED, PackageManager.DONT_KILL_APP)
        for ((a, _, _) in ALL) if (a != alias) pm.setComponentEnabledSetting(component(ctx, a),
            PackageManager.COMPONENT_ENABLED_STATE_DISABLED, PackageManager.DONT_KILL_APP)
    }

    /**
     * Asks the launcher to pin an Anima shortcut whose icon is [picture]. Returns false when the launcher cannot pin
     * shortcuts. The picture is cropped square around its upper middle (where a face usually is in a portrait) and
     * set inside the part of the icon a round or rounded-square mask leaves visible.
     */
    fun pinPicture(ctx: Context, picture: Bitmap, id: String): Boolean {
        val sm = ctx.getSystemService(ShortcutManager::class.java) ?: return false
        if (!sm.isRequestPinShortcutSupported) return false
        val side = minOf(picture.width, picture.height)
        val left = (picture.width - side) / 2
        val top = if (picture.height > picture.width) ((picture.height - side) * 0.25f).toInt() else 0
        val out = Bitmap.createBitmap(432, 432, Bitmap.Config.ARGB_8888)
        val c = Canvas(out)
        // the plain margin: the picture's own average colour
        c.drawColor(Bitmap.createScaledBitmap(picture, 1, 1, true).getPixel(0, 0) or Color.BLACK)
        c.drawBitmap(picture, Rect(left, top, left + side, top + side), Rect(51, 51, 381, 381), Paint(Paint.FILTER_BITMAP_FLAG))
        val info = ShortcutInfo.Builder(ctx, "picture-$id")
            .setShortLabel("Anima")
            .setIcon(Icon.createWithAdaptiveBitmap(out))
            .setIntent(Intent(ctx, MainActivity::class.java).setAction(Intent.ACTION_MAIN))
            .build()
        return sm.requestPinShortcut(info, null)
    }
}
