package com.anima.npu

import android.app.Activity
import android.graphics.drawable.ColorDrawable
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Surface
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat

/**
 * The "Paper" look (2026-11-24): soft off-white or pure black pages, white or near-black cards, one blue accent.
 * Black is #000000 so an OLED screen lights nothing where the page is empty. Values: the design's token table.
 */
data class PaperColors(
    val background: Color, val card: Color, val cardLine: Color, val cardShadow: Boolean,
    val text: Color, val muted: Color, val line: Color, val divider: Color,
    val accent: Color, val onAccent: Color, val link: Color, val soft: Color, val softText: Color, val badge: Color,
    val selected: Color, val onSelected: Color, val well: Color, val segmentOn: Color,
    val switchOff: Color, val knobOff: Color, val bad: Color, val badSoft: Color, val dash: Color,
)

private val LightPaper = PaperColors(
    background = Color(0xFFF4F5F7), card = Color(0xFFFFFFFF), cardLine = Color.Transparent, cardShadow = true,
    text = Color(0xFF15171C), muted = Color(0xFF5B616E), line = Color(0xFFDDE1E8), divider = Color(0xFFEEF0F3),
    accent = Color(0xFF3B5BDB), onAccent = Color.White, link = Color(0xFF3B5BDB), soft = Color(0xFFE4E8F6),
    softText = Color(0xFF2F4AB8), badge = Color(0xFF3B5BDB), selected = Color(0xFF15171C), onSelected = Color.White,
    well = Color(0xFFF4F5F7), segmentOn = Color(0xFFFFFFFF), switchOff = Color(0xFFC5CAD4), knobOff = Color.White,
    bad = Color(0xFFB42318), badSoft = Color(0xFFFBEAEA), dash = Color(0xFFC5CAD4),
)

private val BlackPaper = PaperColors(
    background = Color(0xFF000000), card = Color(0xFF111214), cardLine = Color(0xFF1F2125), cardShadow = false,
    text = Color(0xFFF2F3F5), muted = Color(0xFF9BA1AD), line = Color(0xFF2C2F35), divider = Color(0xFF1F2125),
    accent = Color(0xFF3B5BDB), onAccent = Color.White, link = Color(0xFF9DB0FF), soft = Color(0xFF161B33),
    softText = Color(0xFFB7C4FF), badge = Color(0xFF5B78F0), selected = Color(0xFFF2F3F5), onSelected = Color.Black,
    well = Color(0xFF0A0A0B), segmentOn = Color(0xFF1F2125), switchOff = Color(0xFF3A3E46), knobOff = Color(0xFFC5CAD4),
    bad = Color(0xFFFF8A80), badSoft = Color(0xFF2A1212), dash = Color(0xFF3A3E46),
)

fun paperColors(black: Boolean): PaperColors = if (black) BlackPaper else LightPaper

/** Paper's colours on Material's roles, so stock components pick them up without per-screen changes. */
fun paperScheme(p: PaperColors, black: Boolean): ColorScheme {
    val base = if (black) darkColorScheme() else lightColorScheme()
    return base.copy(
        primary = p.accent, onPrimary = p.onAccent, primaryContainer = p.soft, onPrimaryContainer = p.softText,
        secondary = p.accent, onSecondary = p.onAccent, secondaryContainer = p.soft, onSecondaryContainer = p.softText,
        tertiary = p.link, onTertiary = p.onAccent, tertiaryContainer = p.soft, onTertiaryContainer = p.softText,
        background = p.background, onBackground = p.text,
        surface = p.background, onSurface = p.text, surfaceVariant = p.card, onSurfaceVariant = p.muted,
        surfaceTint = Color.Transparent, inverseSurface = p.selected, inverseOnSurface = p.onSelected, inversePrimary = p.link,
        surfaceBright = p.card, surfaceDim = p.background,
        surfaceContainerLowest = p.background, surfaceContainerLow = p.card, surfaceContainer = p.card,
        surfaceContainerHigh = p.card, surfaceContainerHighest = p.card,
        outline = p.line, outlineVariant = p.divider,
        error = p.bad, onError = p.onAccent, errorContainer = p.badSoft, onErrorContainer = p.bad,
        scrim = Color.Black,
    )
}

val LocalPaper = staticCompositionLocalOf { LightPaper }

private fun jakarta(w: Int) = Font(R.font.plus_jakarta_sans, FontWeight(w),
    variationSettings = FontVariation.Settings(FontVariation.weight(w)))

/** Plus Jakarta Sans, bundled (res/font, SIL Open Font License; licence text in assets/licences). */
val PaperFont = FontFamily(jakarta(400), jakarta(500), jakarta(600), jakarta(700))

private fun paperTypography(): Typography {
    val t = Typography()
    fun TextStyle.f() = copy(fontFamily = PaperFont)
    return Typography(
        displayLarge = t.displayLarge.f(), displayMedium = t.displayMedium.f(), displaySmall = t.displaySmall.f(),
        headlineLarge = t.headlineLarge.f(), headlineMedium = t.headlineMedium.f(),
        headlineSmall = t.headlineSmall.f().copy(fontSize = 26.sp, lineHeight = 32.sp, fontWeight = FontWeight.Bold, letterSpacing = (-0.4).sp),
        titleLarge = t.titleLarge.f().copy(fontWeight = FontWeight.Bold), titleMedium = t.titleMedium.f().copy(fontWeight = FontWeight.SemiBold),
        titleSmall = t.titleSmall.f().copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold),
        bodyLarge = t.bodyLarge.f().copy(fontSize = 15.sp), bodyMedium = t.bodyMedium.f(), bodySmall = t.bodySmall.f().copy(fontSize = 12.sp),
        labelLarge = t.labelLarge.f().copy(fontWeight = FontWeight.SemiBold), labelMedium = t.labelMedium.f(), labelSmall = t.labelSmall.f(),
    )
}

private val PaperShapes = Shapes(
    extraSmall = RoundedCornerShape(8.dp), small = RoundedCornerShape(12.dp), medium = RoundedCornerShape(20.dp),
    large = RoundedCornerShape(28.dp), extraLarge = RoundedCornerShape(28.dp),
)

@Composable
fun PaperTheme(black: Boolean, content: @Composable () -> Unit) {
    val p = paperColors(black)
    val view = LocalView.current
    if (!view.isInEditMode) SideEffect {
        val window = (view.context as Activity).window
        // older Androids colour the bars themselves; from Android 15 the page draws behind them (the Surface below)
        @Suppress("DEPRECATION") run { window.statusBarColor = p.background.toArgb(); window.navigationBarColor = p.background.toArgb() }
        window.setBackgroundDrawable(ColorDrawable(p.background.toArgb()))
        WindowCompat.getInsetsController(window, view).apply {
            isAppearanceLightStatusBars = !black
            isAppearanceLightNavigationBars = !black
        }
    }
    CompositionLocalProvider(LocalPaper provides p) {
        MaterialTheme(colorScheme = paperScheme(p, black), typography = paperTypography(), shapes = PaperShapes) {
            Surface(Modifier.fillMaxSize(), color = p.background, contentColor = p.text) { content() }
        }
    }
}
