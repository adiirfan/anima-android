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

/** The app's settings: Ling's memory use and the usage bar. */
@Composable
fun SettingsScreen() {
    val ctx = LocalContext.current
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Chat model and memory use", style = MaterialTheme.typography.titleSmall)
        // (setting, label, what it means): memory and speed measured on the iQOO 13, 2026-10-09
        val choices = listOf(
            Triple(0, "Fastest", "Ling, with everything on the AI chip: about 4.9 GB of memory while it runs, " +
                "writes about 31 words a second."),
            Triple(48, "Balanced", "Ling keeps 48 of each layer's 128 experts on the AI chip and reads the rest from " +
                "storage as needed: about 2.6 GB, writes about 15 words a second."),
            Triple(32, "Low", "Ling keeps 32 experts per layer on the AI chip: about 2.3 GB, writes about 14 " +
                "words a second and reads long messages more slowly."),
            Triple(LocalLing.SMALL, "Lowest", "A much smaller model, Qwen3.5 0.8B, instead of Ling: about 1.4 GB, " +
                "writes about 25 words a second. Less clever: it sometimes answers in words when you asked for a " +
                "picture, so ask again or say \"draw\". A 0.8 GB download, offered on the Chat tab."),
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
        if (slots > 0) {
            Text("Balanced and Low keep a 3.9 GB copy of Ling's experts next to the model, made the first " +
                "time Ling starts (about a minute). Choosing Fastest deletes it.", style = MaterialTheme.typography.bodySmall)
        }
        Text("Takes effect with the next chat message.", style = MaterialTheme.typography.bodySmall)

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
        Text("Your launcher may take a few seconds to show the new icon. To use any picture instead, open it in the " +
            "Gallery and tap \"Add to home screen as an icon\".", style = MaterialTheme.typography.bodySmall)

        HorizontalDivider()

        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Show usage stats at the top", style = MaterialTheme.typography.titleSmall)
                Text("Processor, graphics, AI chip, power, temperature and memory.", style = MaterialTheme.typography.bodySmall)
            }
            Switch(StatsBar.shown(ctx), { StatsBar.set(ctx, it) })
        }
    }
}
