package com.anima.npu

import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** The app's settings, grouped as in the Paper design: Models, Look, Drawing. */
@Composable
fun SettingsScreen(onSetup: () -> Unit) {
    val ctx = LocalContext.current
    val p = LocalPaper.current
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {

        SectionLabel("Models")
        PaperCard(padding = PaddingValues(horizontal = 16.dp)) {
            val onPhone = remember { ModelStore.ALL.count { it.active(ctx).installed(ctx) != null } }
            PaperRow(Modifier.clickable(onClick = onSetup)) {
                Column(Modifier.weight(1f)) {
                    RowTitle("Model management")
                    Note(if (onPhone == 1) "1 model on this phone" else "$onPhone models on this phone")
                    if (!Chip.supported && !Chip.animaOnly) Note("${Chip.label}: no AI chip support, models run on the graphics chip.")
                }
                Icon(AppIcons.Chevron, null, tint = p.muted)
            }

            // (setting, label, what it means): memory measured on the iQOO 13, 2026-10-09; Balanced and Low writing speeds
            // on the S25 Edge, warm, after the 2026-10-10 streaming changes (were 15 and 14)
            val choices = listOf(
                Triple(0, "Fastest", "Ling · 4.9 GB · 31 words/s"),
                Triple(48, "Balanced", "Ling · 2.6 GB · 21 words/s"),
                Triple(32, "Low", "Ling · 2.3 GB · 19 words/s"),
                Triple(LocalLing.SMALL, "Lowest", "Qwen3.5 0.8B · 1.4 GB · 25 words/s · less clever"),
            )
            var slots by remember { mutableIntStateOf(LocalLing.expertSlots(ctx)) }
            Block {
                RowTitle("Chat model")
                Segmented(choices.map { it.second }, choices.indexOfFirst { it.first == slots }) { i ->
                    val n = choices[i].first
                    if (slots != n) { slots = n; LocalLing.setExpertSlots(ctx, n) }
                }
                Note(choices.firstOrNull { it.first == slots }?.third ?: "")
            }
            HorizontalDivider(color = p.divider)

            val engines = listOf("Automatic", "OpenCL", "Vulkan")
            var engine by remember { mutableIntStateOf(GpuEngine.engineChoice(ctx)) }
            Block {
                RowTitle("Graphics engine")
                Segmented(engines, engine) { i -> if (engine != i) { engine = i; GpuEngine.setEngineChoice(ctx, i) } }
                Note(when (engine) {
                    1 -> "OpenCL draws every model on the graphics chip."
                    2 -> "Vulkan draws ${GpuEngine.VULKAN_MODELS}; other models use OpenCL. If Vulkan fails on this phone, OpenCL takes over."
                    else -> "Vulkan draws ${GpuEngine.VULKAN_MODELS} on the Snapdragon 8 Elite Gen 5, about 1.5× faster; everything else uses OpenCL."
                })
            }
        }

        SectionLabel("Look")
        PaperCard(padding = PaddingValues(horizontal = 16.dp)) {
            Block {
                RowTitle("Theme")
                Segmented(listOf("Phone", "Light", "Black"), Look.mode.ordinal) { Look.set(ctx, ThemeMode.values()[it]) }
                Note(when (Look.mode) {
                    ThemeMode.PHONE -> "Follows the phone's dark mode."
                    ThemeMode.LIGHT -> "Always light."
                    ThemeMode.BLACK -> "Always black."
                })
            }
            HorizontalDivider(color = p.divider)
            var icon by remember { mutableStateOf(AppIcon.current(ctx)) }
            Block {
                RowTitle("App icon")
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    AppIcon.ALL.forEach { (alias, name, res) ->
                        val shape = RoundedCornerShape(14.dp)
                        Image(painterResource(res), name, Modifier.size(52.dp).clip(shape)
                            .border(if (icon == alias) 2.dp else 1.dp, if (icon == alias) p.accent else p.line, shape)
                            .clickable { if (icon != alias) { icon = alias; AppIcon.set(ctx, alias) } },
                            contentScale = ContentScale.Crop)
                    }
                }
            }
        }

        SectionLabel("Drawing")
        PaperCard(padding = PaddingValues(horizontal = 16.dp)) {
            if (Chip.supported) PaperRow {
                Text("Upscale every new picture 4×", Modifier.weight(1f), fontSize = 15.sp, fontWeight = FontWeight.Medium)
                Switch(Upscaler.auto(ctx), { Upscaler.setAuto(ctx, it) }, colors = paperSwitchColors())
            }
            PaperRow(divider = false) {
                Text("Show usage stats at the top", Modifier.weight(1f), fontSize = 15.sp, fontWeight = FontWeight.Medium)
                Switch(StatsBar.shown(ctx), { StatsBar.set(ctx, it) }, colors = paperSwitchColors())
            }
        }
    }
}

@Composable
private fun RowTitle(text: String) = Text(text, fontSize = 15.sp, fontWeight = FontWeight.Medium)

@Composable
private fun Note(text: String) = Text(text, style = MaterialTheme.typography.bodySmall, color = LocalPaper.current.muted)

/** A setting with its control under its name, inside a card. */
@Composable
private fun Block(content: @Composable ColumnScope.() -> Unit) =
    Column(Modifier.fillMaxWidth().padding(vertical = 14.dp), verticalArrangement = Arrangement.spacedBy(10.dp), content = content)
