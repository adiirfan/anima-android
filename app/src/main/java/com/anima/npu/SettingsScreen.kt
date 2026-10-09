package com.anima.npu

import androidx.compose.foundation.layout.Arrangement
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
        Text("Settings", style = MaterialTheme.typography.titleLarge)

        Text("Ling's memory use (chat)", style = MaterialTheme.typography.titleSmall)
        // (pool size, label, what it means): measured on the S25 Edge, 2026-10-09; 48 estimated
        val choices = listOf(
            Triple(0, "Fastest", "Everything stays on the AI chip: about 4.5 GB of memory locked while Ling runs, " +
                "writes about 25 words a second."),
            Triple(48, "Balanced", "Ling keeps 48 of each layer's 128 experts on the AI chip and reads the rest from " +
                "storage as needed: about 2.3 GB, a little slower to write."),
            Triple(32, "Low memory", "Ling keeps 32 experts per layer on the AI chip: about 1.9 GB, writes about 13 " +
                "words a second and reads long messages more slowly. Leaves the most room for other apps."),
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
            Text("Balanced and Low memory keep a 3.9 GB copy of Ling's experts next to the model, made the first " +
                "time Ling starts (about a minute). Choosing Fastest deletes it.", style = MaterialTheme.typography.bodySmall)
        }
        Text("Takes effect with the next chat message.", style = MaterialTheme.typography.bodySmall)

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
