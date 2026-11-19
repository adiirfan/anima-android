package com.anima.npu

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/*
 * Chat text drawn the way sysmon's character chat draws it (2026-11-18): words said aloud in "quotes" in a warm
 * orange, *asterisk* parts (actions, thoughts) in a soft italic colour, **bold** as bold. On top of that the plain
 * answer shapes a chat model writes: headings, bullet and numbered lists, > quotes, tables, `inline code` and fenced
 * code in its own box. A mark that is never closed (a reply still streaming in) stays plain text, so nothing flickers
 * between styles halfway through a word.
 */

private sealed interface Block {
    data class Para(val text: String) : Block
    data class Heading(val level: Int, val text: String) : Block
    data class Item(val marker: String, val indent: Int, val text: String) : Block
    data class Quote(val text: String) : Block
    data class Code(val lang: String, val code: String) : Block
    data class Table(val rows: List<List<String>>) : Block
    data object Rule : Block
}

private val HEADING = Regex("""^(#{1,6})\s+(.*)$""")
private val BULLET = Regex("""^(\s*)[-*+•]\s+(.*)$""")
private val NUMBER = Regex("""^(\s*)(\d{1,3})[.)]\s+(.*)$""")
private val RULE = Regex("""^\s*([-*_])(\s*\1){2,}\s*$""")
private val TABLE_SEP = Regex("""^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$""")

private fun cells(line: String) = line.trim().removePrefix("|").removeSuffix("|").split("|").map { it.trim() }

private fun parse(src: String): List<Block> {
    val out = mutableListOf<Block>()
    val lines = src.replace("\r\n", "\n").split("\n")
    val para = StringBuilder()
    fun flush() { if (para.isNotBlank()) out += Block.Para(para.toString().trimEnd()); para.clear() }
    var i = 0
    while (i < lines.size) {
        val line = lines[i]
        val t = line.trim()
        when {
            t.startsWith("```") -> {
                flush()
                val lang = t.removePrefix("```").trim()
                val code = StringBuilder()
                i++
                // no closing fence yet (still streaming): everything after the opening one is code
                while (i < lines.size && !lines[i].trim().startsWith("```")) { code.append(lines[i]).append('\n'); i++ }
                out += Block.Code(lang, code.toString().trimEnd('\n'))
            }
            t.isEmpty() -> flush()
            RULE.matches(line) && !BULLET.matches(line) || t == "***" || t == "---" -> { flush(); out += Block.Rule }
            HEADING.matches(t) -> { flush(); val m = HEADING.find(t)!!; out += Block.Heading(m.groupValues[1].length, m.groupValues[2]) }
            t.startsWith("|") && i + 1 < lines.size && TABLE_SEP.matches(lines[i + 1]) -> {
                flush()
                val rows = mutableListOf(cells(t))
                i += 2
                while (i < lines.size && lines[i].trim().startsWith("|")) { rows += cells(lines[i]); i++ }
                out += Block.Table(rows)
                continue
            }
            t.startsWith(">") -> {
                flush()
                val q = StringBuilder()
                while (i < lines.size && lines[i].trim().startsWith(">")) {
                    q.append(lines[i].trim().removePrefix(">").removePrefix(" ")).append('\n'); i++
                }
                out += Block.Quote(q.toString().trimEnd())
                continue
            }
            BULLET.matches(line) -> {
                flush(); val m = BULLET.find(line)!!
                out += Block.Item("•", m.groupValues[1].length / 2, m.groupValues[2])
            }
            NUMBER.matches(line) -> {
                flush(); val m = NUMBER.find(line)!!
                out += Block.Item(m.groupValues[2] + ".", m.groupValues[1].length / 2, m.groupValues[3])
            }
            else -> {
                // a line under a list item that is indented belongs to that item
                val last = out.lastOrNull()
                if (para.isEmpty() && last is Block.Item && line.startsWith("  ")) out[out.size - 1] = last.copy(text = last.text + "\n" + t)
                else { if (para.isNotEmpty()) para.append('\n'); para.append(line) }
            }
        }
        i++
    }
    flush()
    return out
}

/** The orange sysmon uses for speech: lighter on a dark screen so it stays easy to read. */
fun speechColour(scheme: ColorScheme): Color =
    if (scheme.background.luminance() < 0.5f) Color(0xFFF0A35E) else Color(0xFFB35C00)

private val INLINE = Regex(
    """`[^`\n]+`""" +                           // inline code
    """|\*\*[^*\n]+?\*\*|__[^_\n]+?__""" +      // bold
    """|~~[^~\n]+?~~""" +                       // struck out
    """|\*[^*\n]+?\*|(?<![\w])_[^_\n]+?_(?![\w])""" + // italic: actions and thoughts
    """|"[^"\n]*"|“[^”\n]*”"""                  // speech
)

private class Styles(val speech: Color, val thought: Color, val codeBg: Color)

private fun AnnotatedString.Builder.inline(text: String, s: Styles) {
    var from = 0
    for (m in INLINE.findAll(text)) {
        if (m.range.first > from) append(text.substring(from, m.range.first))
        val v = m.value
        when {
            v.startsWith("`") -> withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = s.codeBg, fontSize = 0.92.em())) {
                append(v.substring(1, v.length - 1))
            }
            v.startsWith("**") || v.startsWith("__") -> withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { inline(v.substring(2, v.length - 2), s) }
            v.startsWith("~~") -> withStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)) { inline(v.substring(2, v.length - 2), s) }
            v.startsWith("*") || v.startsWith("_") -> withStyle(SpanStyle(fontStyle = FontStyle.Italic, color = s.thought)) {
                inline(v.substring(1, v.length - 1), s)
            }
            // the quotes stay: they are part of how speech reads
            else -> withStyle(SpanStyle(color = s.speech)) { append(v) }
        }
        from = m.range.last + 1
    }
    if (from < text.length) append(text.substring(from))
}

private fun Double.em() = androidx.compose.ui.unit.TextUnit(this.toFloat(), androidx.compose.ui.unit.TextUnitType.Em)

/**
 * [text] formatted. [fg] is the colour of plain words, [fontSize]/[lineHeight] their size. The caller decides about
 * selection (wrap it in a SelectionContainer) and the bubble around it.
 */
@Composable
fun RichText(text: String, fg: Color = Color.Unspecified, fontSize: TextUnit = 15.sp, lineHeight: TextUnit = 21.sp,
             modifier: Modifier = Modifier) {
    val scheme = MaterialTheme.colorScheme
    val blocks = remember(text) { parse(text) }
    val dark = scheme.background.luminance() < 0.5f
    val st = Styles(speechColour(scheme), scheme.tertiary, if (dark) Color(0x33FFFFFF) else Color(0x1A000000))
    val fmt = { s: String -> buildAnnotatedString { inline(s, st) } }
    Column(modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        for (b in blocks) when (b) {
            is Block.Para -> Text(fmt(b.text), color = fg, fontSize = fontSize, lineHeight = lineHeight)
            is Block.Heading -> Text(fmt(b.text), Modifier.padding(top = 4.dp), color = fg, fontWeight = FontWeight.SemiBold,
                fontSize = when (b.level) { 1 -> fontSize * 1.35f; 2 -> fontSize * 1.2f; else -> fontSize * 1.07f },
                lineHeight = lineHeight * 1.2f)
            is Block.Item -> Row(Modifier.padding(start = (b.indent * 16).dp)) {
                Text(b.marker, Modifier.width(if (b.marker == "•") 16.dp else 24.dp), color = scheme.onSurfaceVariant,
                    fontSize = fontSize, lineHeight = lineHeight)
                Text(fmt(b.text), color = fg, fontSize = fontSize, lineHeight = lineHeight)
            }
            is Block.Quote -> Row(Modifier.height(IntrinsicSize.Min)) {
                Box(Modifier.width(3.dp).fillMaxHeight().clip(RoundedCornerShape(2.dp)).background(scheme.outline))
                Text(fmt(b.text), Modifier.padding(start = 10.dp), color = scheme.onSurfaceVariant, fontSize = fontSize,
                    lineHeight = lineHeight, fontStyle = FontStyle.Italic)
            }
            is Block.Code -> CodeBox(b.lang, b.code, dark)
            is Block.Table -> TableBox(b.rows, fmt, fg, fontSize)
            Block.Rule -> HorizontalDivider(Modifier.padding(vertical = 4.dp), color = scheme.outlineVariant)
        }
    }
}

@Composable
private fun CodeBox(lang: String, code: String, dark: Boolean) {
    val ctx = LocalContext.current
    val scheme = MaterialTheme.colorScheme
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(if (dark) Color(0x66000000) else Color(0x0F000000))) {
        Row(Modifier.fillMaxWidth().padding(start = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(lang.ifEmpty { "code" }, Modifier.weight(1f), fontSize = 11.5.sp, color = scheme.onSurfaceVariant)
            IconButton(onClick = {
                (ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("code", code))
            }, Modifier.size(32.dp)) { Icon(AppIcons.Copy, "Copy code", Modifier.size(15.dp), tint = scheme.onSurfaceVariant) }
        }
        Text(code, Modifier.horizontalScroll(rememberScrollState()).padding(start = 12.dp, end = 12.dp, bottom = 10.dp),
            fontFamily = FontFamily.Monospace, fontSize = 13.sp, lineHeight = 18.sp, softWrap = false)
    }
}

@Composable
private fun TableBox(rows: List<List<String>>, fmt: (String) -> AnnotatedString, fg: Color, fontSize: TextUnit) {
    val scheme = MaterialTheme.colorScheme
    val n = rows.maxOf { it.size }
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(scheme.surface.copy(alpha = 0.35f))) {
        rows.forEachIndexed { r, row ->
            if (r > 0) HorizontalDivider(color = scheme.outlineVariant.copy(alpha = 0.6f))
            Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp)) {
                for (c in 0 until n) Text(fmt(row.getOrElse(c) { "" }), Modifier.weight(1f).padding(end = 6.dp), color = fg,
                    fontSize = fontSize * 0.93f, lineHeight = fontSize * 1.3f, fontWeight = if (r == 0) FontWeight.SemiBold else null)
            }
        }
    }
}
