package com.anima.npu

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.widthIn
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.unit.dp

// Unfolded foldables and tablets (2026-11-18): from 600 dp across the menu stays open at the side, Create shows its
// settings and the picture side by side, and text pages keep a readable width instead of running across the screen.
// The folded cover screen and ordinary phones are under 600 dp and keep the one-column layout.
@Composable
fun isWide(): Boolean = LocalConfiguration.current.screenWidthDp >= 600

/** Content no wider than [max], centred, on wide screens; unchanged on narrow ones. */
@Composable
fun Readable(max: Int = 760, content: @Composable () -> Unit) {
    if (!isWide()) { content(); return }
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
        Box(Modifier.widthIn(max = max.dp).fillMaxSize()) { content() }
    }
}
