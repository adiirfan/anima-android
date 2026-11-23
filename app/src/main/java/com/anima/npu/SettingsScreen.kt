package com.anima.npu

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.clickable
import androidx.compose.foundation.border
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

/** The app's settings: which models to download, Ling's memory use, the graphics engine and the usage bar. */
@Composable
fun SettingsScreen(onSetup: () -> Unit) {
    val ctx = LocalContext.current
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Models", style = MaterialTheme.typography.titleSmall)
        if (!Chip.supported && !Chip.animaOnly) Text("${Chip.label}: no AI chip support, models run on the graphics chip.",
            style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = onSetup) { Text("Model Management") }

        HorizontalDivider()

        Text("Chat model", style = MaterialTheme.typography.titleSmall)
        // (setting, label, what it means): memory measured on the iQOO 13, 2026-10-09; Balanced and Low writing speeds
        // on the S25 Edge, warm, after the 2026-10-10 streaming changes (were 15 and 14)
        val choices = listOf(
            Triple(0, "Fastest", "Ling · 4.9 GB · 31 words/s"),
            Triple(48, "Balanced", "Ling · 2.6 GB · 21 words/s"),
            Triple(32, "Low", "Ling · 2.3 GB · 19 words/s"),
            Triple(LocalLing.SMALL, "Lowest", "Qwen3.5 0.8B · 1.4 GB · 25 words/s · less clever"),
        )
        var slots by remember { mutableIntStateOf(LocalLing.expertSlots(ctx)) }
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            choices.forEachIndexed { i, (n, label, _) ->
                SegmentedButton(selected = slots == n, onClick = {
                    if (slots != n) { slots = n; LocalLing.setExpertSlots(ctx, n) }
                }, shape = SegmentedButtonDefaults.itemShape(i, choices.size)) { Text(label, maxLines = 1) }
            }
        }
        Text(choices.first { it.first == slots }.third, style = MaterialTheme.typography.bodySmall)

        HorizontalDivider()

        Text("Graphics engine", style = MaterialTheme.typography.titleSmall)
        val engines = listOf("Automatic", "OpenCL", "Vulkan")
        var engine by remember { mutableIntStateOf(GpuEngine.engineChoice(ctx)) }
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            engines.forEachIndexed { i, label ->
                SegmentedButton(selected = engine == i, onClick = {
                    if (engine != i) { engine = i; GpuEngine.setEngineChoice(ctx, i) }
                }, shape = SegmentedButtonDefaults.itemShape(i, engines.size)) { Text(label, maxLines = 1) }
            }
        }
        Text(when (engine) {
            1 -> "OpenCL draws every model on the graphics chip."
            2 -> "Vulkan draws ${GpuEngine.VULKAN_MODELS}; other models use OpenCL. If Vulkan fails on this phone, OpenCL takes over."
            else -> "Vulkan draws ${GpuEngine.VULKAN_MODELS} on the Snapdragon 8 Elite Gen 5, about 1.5× faster; everything else uses OpenCL."
        }, style = MaterialTheme.typography.bodySmall)

        HorizontalDivider()

        Text("App icon", style = MaterialTheme.typography.titleSmall)
        var icon by remember { mutableStateOf(AppIcon.current(ctx)) }
        Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            AppIcon.ALL.forEach { (alias, name, res) ->
                Image(painterResource(res), name, Modifier.size(60.dp).clip(CircleShape)
                    .border(if (icon == alias) 3.dp else 0.dp, MaterialTheme.colorScheme.primary, CircleShape)
                    .clickable { if (icon != alias) { icon = alias; AppIcon.set(ctx, alias) } },
                    contentScale = ContentScale.Crop)
            }
        }

        HorizontalDivider()

        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Show usage stats at the top", style = MaterialTheme.typography.titleSmall)
            }
            Switch(StatsBar.shown(ctx), { StatsBar.set(ctx, it) })
        }

        if (Chip.supported) Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Upscale every new picture 4×", style = MaterialTheme.typography.titleSmall)
            }
            Switch(Upscaler.auto(ctx), { Upscaler.setAuto(ctx, it) })
        }

    }
}
