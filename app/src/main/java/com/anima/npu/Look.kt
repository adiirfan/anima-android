package com.anima.npu

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/** Phone follows the system's dark mode; Light and Black are fixed. Black is the pure-black (OLED) Paper look. */
enum class ThemeMode { PHONE, LIGHT, BLACK }

/** The chosen theme, kept in the "anima" preferences under "theme". Changing [mode] re-themes the app at once. */
object Look {
    var mode by mutableStateOf(ThemeMode.PHONE)
        private set

    fun parse(s: String?): ThemeMode = when (s) {
        "light" -> ThemeMode.LIGHT
        "black" -> ThemeMode.BLACK
        else -> ThemeMode.PHONE
    }

    fun key(m: ThemeMode): String = m.name.lowercase()

    fun isBlack(m: ThemeMode, systemDark: Boolean): Boolean = when (m) {
        ThemeMode.PHONE -> systemDark
        ThemeMode.LIGHT -> false
        ThemeMode.BLACK -> true
    }

    private fun prefs(ctx: Context) = ctx.getSharedPreferences("anima", Context.MODE_PRIVATE)

    fun load(ctx: Context) { mode = parse(prefs(ctx).getString("theme", null)) }

    fun set(ctx: Context, m: ThemeMode) {
        mode = m
        prefs(ctx).edit().putString("theme", key(m)).apply()
    }
}
