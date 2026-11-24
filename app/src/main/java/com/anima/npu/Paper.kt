package com.anima.npu

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.SwitchColors
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.style.LineHeightStyle
import androidx.compose.ui.text.PlatformTextStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.draw.drawBehind
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.clickable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// The Paper pieces every page is built from (2026-11-24). Sizes follow the mockups: cards 20 dp corners,
// pills 32 dp tall and fully round, the big action button 56 dp.

private val CardShape = RoundedCornerShape(20.dp)

/** A rounded card: a faint shadow on Light, a thin outline on Black (shadows don't show on black). */
@Composable
fun PaperCard(modifier: Modifier = Modifier, padding: PaddingValues = PaddingValues(16.dp),
              content: @Composable ColumnScope.() -> Unit) {
    val p = LocalPaper.current
    Column(
        modifier
            .then(if (p.cardShadow) Modifier.shadow(1.dp, CardShape, ambientColor = Color(0x0F15171C), spotColor = Color(0x0F15171C)) else Modifier)
            .clip(CardShape)
            .background(p.card)
            .border(1.dp, p.cardLine, CardShape)
            .padding(padding),
        content = content,
    )
}

/** The small muted heading above a card ("Models", "Today"). */
@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier) =
    Text(text, modifier.padding(start = 4.dp, end = 4.dp, top = 4.dp), style = MaterialTheme.typography.titleSmall,
        color = LocalPaper.current.muted)

/** A round choice or action button. [selected] draws it filled (the current choice). */
@Composable
fun Pill(text: String, selected: Boolean = false, enabled: Boolean = true, modifier: Modifier = Modifier,
         onClick: () -> Unit) {
    val p = LocalPaper.current
    Surface(
        onClick = onClick, enabled = enabled, shape = CircleShape,
        color = if (selected) p.selected else p.card,
        contentColor = if (selected) p.onSelected else p.text,
        border = if (selected) null else BorderStroke(1.dp, p.line),
        modifier = modifier.heightIn(min = 32.dp).semantics { this.selected = selected },
    ) {
        Row(Modifier.padding(horizontal = 12.dp).defaultMinSize(minHeight = 32.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(text, fontSize = 13.sp, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
                maxLines = 1, overflow = TextOverflow.Ellipsis,
                color = if (enabled) Color.Unspecified else p.muted)
        }
    }
}

/** The page's main action (Draw): full round, accent coloured, 56 dp tall. */
@Composable
fun BigButton(text: String, icon: ImageVector? = null, enabled: Boolean = true, busy: Boolean = false,
              modifier: Modifier = Modifier, onClick: () -> Unit) {
    val p = LocalPaper.current
    Button(onClick, modifier.height(56.dp), enabled = enabled, shape = CircleShape,
        colors = ButtonDefaults.buttonColors(containerColor = p.accent, contentColor = p.onAccent,
            disabledContainerColor = p.line, disabledContentColor = p.muted)) {
        if (busy) { CircularProgressIndicator(Modifier.size(18.dp), color = p.onAccent, strokeWidth = 2.dp); Spacer(Modifier.width(8.dp)) }
        else if (icon != null) { Icon(icon, null, Modifier.size(18.dp)); Spacer(Modifier.width(8.dp)) }
        Text(text, fontSize = 17.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun paperSwitchColors(): SwitchColors {
    val p = LocalPaper.current
    return SwitchDefaults.colors(
        checkedThumbColor = Color.White, checkedTrackColor = p.accent, checkedBorderColor = Color.Transparent,
        uncheckedThumbColor = p.knobOff, uncheckedTrackColor = p.switchOff, uncheckedBorderColor = Color.Transparent,
    )
}

/** A row of equal choices in a sunken well, the chosen one raised (Theme: Phone / Light / Black). */
@Composable
fun Segmented(options: List<String>, selected: Int, modifier: Modifier = Modifier, onSelect: (Int) -> Unit) {
    val p = LocalPaper.current
    val outer = RoundedCornerShape(14.dp)
    val inner = RoundedCornerShape(10.dp)
    Row(
        modifier.fillMaxWidth().clip(outer).background(p.well).border(1.dp, p.cardLine, outer).padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        options.forEachIndexed { i, label ->
            val on = i == selected
            Surface(
                onClick = { onSelect(i) }, shape = inner,
                color = if (on) p.segmentOn else Color.Transparent,
                contentColor = if (on) p.text else p.muted,
                shadowElevation = if (on && p.cardShadow) 1.dp else 0.dp,
                modifier = Modifier.weight(1f).height(38.dp).semantics { this.selected = on; role = Role.Tab },
            ) {
                Row(horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
                    Text(label, fontSize = 13.sp, fontWeight = if (on) FontWeight.SemiBold else FontWeight.Medium)
                }
            }
        }
    }
}

/** One line inside a card, at least 56 dp tall, with a divider under it unless it is the last. */
@Composable
fun PaperRow(modifier: Modifier = Modifier, divider: Boolean = true, content: @Composable RowScope.() -> Unit) {
    Column(modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).padding(vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically, content = content)
        if (divider) HorizontalDivider(color = LocalPaper.current.divider)
    }
}

/** A choice of one among a few, as a row of pills that wraps when it runs out of room. [enabledAt]: which can be picked. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun PillChoices(options: List<String>, selected: Int, enabled: Boolean = true, modifier: Modifier = Modifier,
                enabledAt: (Int) -> Boolean = { true }, onSelect: (Int) -> Unit) {
    FlowRow(modifier, horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        options.forEachIndexed { i, label ->
            Pill(label, selected = i == selected, enabled = enabled && enabledAt(i)) { onSelect(i) }
        }
    }
}

/** The quiet dashed button for an optional extra ("Guide picture"). */
@Composable
fun DashedButton(text: String, icon: ImageVector? = null, enabled: Boolean = true, onClick: () -> Unit) {
    val p = LocalPaper.current
    val shape = RoundedCornerShape(10.dp)
    Row(
        Modifier.heightIn(min = 36.dp).clip(shape).background(p.well)
            .drawBehind {
                drawRoundRect(p.dash, cornerRadius = CornerRadius(10.dp.toPx()),
                    style = Stroke(1.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx()))))
            }
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        if (icon != null) Icon(icon, null, Modifier.size(16.dp), tint = p.muted)
        Text(text, fontSize = 13.sp, fontWeight = FontWeight.Medium, color = if (enabled) p.muted else p.line)
    }
}

/** The Queue button at the top of Create: soft blue, with how many pictures wait. */
@Composable
fun QueueChip(waiting: Int, modifier: Modifier = Modifier, onClick: () -> Unit) {
    val p = LocalPaper.current
    Surface(onClick = onClick, shape = CircleShape, color = p.soft, contentColor = p.softText, modifier = modifier.height(36.dp)) {
        Row(Modifier.padding(horizontal = 14.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Queue", fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
            if (waiting > 0) Box(Modifier.defaultMinSize(minWidth = 20.dp).height(20.dp).clip(CircleShape).background(p.badge)
                .padding(horizontal = 6.dp), contentAlignment = Alignment.Center) {
                Text("$waiting", style = BadgeNumber, color = Color.White)
            }
        }
    }
}

/** A number inside a small round badge: the font's extra space above and below is trimmed so it sits in the middle. */
val BadgeNumber = TextStyle(
    fontFamily = PaperFont, fontSize = 12.sp, lineHeight = 12.sp, fontWeight = FontWeight.Bold,
    platformStyle = PlatformTextStyle(includeFontPadding = false),
    lineHeightStyle = LineHeightStyle(LineHeightStyle.Alignment.Center, LineHeightStyle.Trim.Both),
)
