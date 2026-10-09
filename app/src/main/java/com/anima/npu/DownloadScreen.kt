package com.anima.npu

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

private fun gb(bytes: Long) = "%.1f GB".format(bytes / 1e9)

/**
 * First-run download, and switching between the standard and low-memory drawing model later.
 * [current] is the installed variant id (null on first run); [onBack] only exists when switching.
 * The work itself runs in [Installer], so leaving this screen does not stop it; the app picks up the
 * finished install from [Installer.finished].
 */
@Composable
fun DownloadScreen(
    store: ModelStore, current: String?, beforeSwap: () -> Unit, onBack: (() -> Unit)?, update: Boolean = false,
) {
    val ctx = LocalContext.current
    var manifest by remember { mutableStateOf(store.cachedManifest(ctx)) }
    var choice by remember(manifest) {
        mutableStateOf(if (update) current!! else manifest?.variants?.firstOrNull { it.id != current }?.id ?: "standard")
    }
    var status by remember { mutableStateOf("") }
    val busy = Installer.running

    if (onBack != null) BackHandler(onBack = onBack)
    LaunchedEffect(Unit) {
        // Always refresh the list so updated model files are picked up; fall back to the cached one offline.
        status = "Getting the model list…"
        runCatching { store.fetchManifest(ctx) }
            .onSuccess { manifest = it; status = "" }
            .onFailure { status = if (manifest == null) "Could not reach Hugging Face: ${it.message}" else "" }
    }

    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        if (onBack != null) TextButton(onClick = onBack) { Text("‹ Back") }
        Text((if (current == null) "Download " else if (update) "Update " else "Change ") + store.title,
            style = MaterialTheme.typography.headlineSmall)
        Text("Everything runs on the phone's neural unit; nothing is sent anywhere after this download.",
            style = MaterialTheme.typography.bodySmall)
        val m = manifest
        if (m != null) {
            for (v in m.variants) {
                if (update && v.id != current) continue
                val size = when {
                    update -> store.outdated(ctx, m).sumOf { it.size }
                    current == null -> m.bytes(v.id)
                    else -> v.files.sumOf { it.size }
                }
                val pickable = !busy && (update || v.id != current)
                Card(Modifier.fillMaxWidth().clickable(enabled = pickable) { choice = v.id }) {
                    Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selected = choice == v.id, onClick = { choice = v.id }, enabled = pickable)
                        Column(Modifier.padding(start = 8.dp)) {
                            Text(v.label + if (v.id == current) " (installed)" else "", style = MaterialTheme.typography.titleMedium)
                            Text(v.about, style = MaterialTheme.typography.bodySmall)
                            Text("Download: ${gb(size)}", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
            Text(if (onBack != null) "Use Wi-Fi. You can leave this screen or the app; the download carries on."
                else "Use Wi-Fi. The download continues if you leave the app.", style = MaterialTheme.typography.bodySmall)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { Installer.start(ctx, store, m, choice, beforeSwap) }, enabled = !busy && (update || choice != current)) {
                    Text(if (current == null) "Download" else if (update) "Update" else "Switch")
                }
                if (busy) OutlinedButton(onClick = { Installer.cancel(ctx) }) { Text("Cancel") }
            }
            if (busy) LinearProgressIndicator(progress = { Installer.fraction }, modifier = Modifier.fillMaxWidth())
            if (Installer.status.isNotEmpty()) Text(Installer.status)
        }
        if (status.isNotEmpty()) Text(status)
        Text(store.licence, style = MaterialTheme.typography.bodySmall)
    }
}
