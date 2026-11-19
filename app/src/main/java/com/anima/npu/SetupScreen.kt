package com.anima.npu

import android.os.StatFs
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
import androidx.compose.material3.Checkbox
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope

private fun gb(bytes: Long) = "%.1f GB".format(bytes / 1e9)

/** What each model does, in a line, for the setup list. */
private val ABOUT = mapOf(
    "anima" to "Anime-style pictures from a prompt, in six sizes. The quickest to draw.",
    "moss" to "768×768 pictures from plain sentences, in 8 steps.",
    "fibo" to "Changes your own pictures as you describe. Edits only.",
    "wan" to "Turns one of your pictures into a short moving clip.",
    "qwen" to "768×768 pictures from plain sentences, and edits your pictures.",
    "klein" to "Pictures from plain sentences, and edits your pictures.",
    "ling" to "Chat on the phone that can also add pictures to the queue for you. Uses 2.3 to 4.9 GB of memory.",
    "smallchat" to "A much smaller chat model for phones short of memory. Less clever than Ling.",
)

/** Shown instead of the Create and Chat screens on a phone whose chip has no build of the models. */
@Composable
fun DeviceNotSupported(modifier: Modifier = Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Device not supported", style = MaterialTheme.typography.headlineSmall)
        Text("This phone has ${Chip.label}. ${Chip.SUPPORTED_TEXT} The models are built for those chips' neural units " +
            "and cannot run on other phones.", style = MaterialTheme.typography.bodyMedium)
    }
}

/**
 * First-run setup: tick the picture and chat models to download, then they all download at the same time.
 * [onDone] leaves the screen (finished or skipped); downloads carry on in [Installer] after that.
 */
@Composable
fun SetupScreen(firstRun: Boolean, onDone: () -> Unit) {
    val ctx = LocalContext.current
    if (!firstRun) androidx.activity.compose.BackHandler(onBack = onDone)
    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(if (firstRun) "Welcome to Anima" else "Choose models", style = MaterialTheme.typography.headlineSmall)
        if (ModelStore.ALL.none { it.available }) {
            DeviceNotSupported()
            Button(onClick = onDone) { Text("Continue") }
            return@Column
        }
        Text(if (Chip.supported) "Choose what to download. Everything runs on this phone's neural unit (${Chip.label}); " +
                "nothing is sent anywhere after the download. You can add or remove models later in Settings."
            else "This phone has ${Chip.label}, which has no AI chip the app supports. MOSS-Image (new pictures) and " +
                "Fibo (picture edits) can still run on its graphics chip, more slowly: several minutes a picture. " +
                "Nothing is sent anywhere after the download.",
            style = MaterialTheme.typography.bodySmall)

        val all = ModelStore.SETUP_PICTURE + ModelStore.SETUP_CHAT
        val manifests = remember { mutableStateMapOf<String, ModelStore.Manifest>() }
        var listStatus by remember { mutableStateOf("Getting the model lists…") }
        // nothing ticked until the user picks; Anima is the one most people want, so it starts ticked
        val chosen = remember { mutableStateMapOf((if (Chip.supported) "anima" else "moss") to true) }
        var started by remember { mutableStateOf(Installer.tasks.isNotEmpty()) }

        LaunchedEffect(Unit) {
            // all the lists at once (each is a few kilobytes); a cached one stands in when offline
            val failed = coroutineScope {
                all.filter { it.available }.map { st ->
                    async {
                        val m = runCatching { st.fetchManifest(ctx) }.getOrNull() ?: st.cachedManifest(ctx)
                        if (m != null) manifests[st.id] = m
                        m == null
                    }
                }.awaitAll().count { it }
            }
            listStatus = if (failed > 0) "Could not get $failed of the model lists from Hugging Face. Check the connection, " +
                "or try again later." else ""
        }

        fun sizeOf(st: ModelStore) = manifests[st.id]?.let { m -> m.bytes(m.variants.first().id) }
        val free = remember { runCatching { StatFs(ctx.getExternalFilesDir(null)!!.path).availableBytes }.getOrDefault(0L) }

        @Composable
        fun row(st: ModelStore) {
            val have = st.installed(ctx) != null
            val size = sizeOf(st)
            val can = st.available && !have && size != null && !started
            Card(Modifier.fillMaxWidth().clickable(enabled = can) { chosen[st.id] = chosen[st.id] != true }) {
                Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(checked = can && chosen[st.id] == true || have, onCheckedChange = { chosen[st.id] = it }, enabled = can)
                    Column(Modifier.padding(start = 8.dp).weight(1f)) {
                        Text(st.title, style = MaterialTheme.typography.titleMedium)
                        Text(ABOUT[st.id] ?: "", style = MaterialTheme.typography.bodySmall)
                        Text(when {
                            !st.available -> "Not available for ${Chip.label} yet"
                            have -> "Downloaded"
                            size == null -> if (listStatus.isEmpty()) "Getting the size…" else "Not available right now"
                            else -> "Download: ${gb(size)}"
                        }, style = MaterialTheme.typography.bodySmall)
                        Installer.task(st.id)?.let { t ->
                            LinearProgressIndicator(progress = { t.fraction }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp))
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(t.status, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                                if (t.running) TextButton(onClick = { Installer.cancel(st.id) }) { Text("Cancel") }
                            }
                        }
                    }
                }
            }
        }

        Text("Pictures and clips", style = MaterialTheme.typography.titleSmall)
        ModelStore.SETUP_PICTURE.forEach { row(it) }
        Text("Chat", style = MaterialTheme.typography.titleSmall)
        ModelStore.SETUP_CHAT.forEach { row(it) }

        val picked = all.filter { it.available && chosen[it.id] == true && it.installed(ctx) == null && sizeOf(it) != null }
        val total = picked.sumOf { sizeOf(it) ?: 0L }
        if (listStatus.isNotEmpty()) Text(listStatus, style = MaterialTheme.typography.bodySmall)
        if (!started) {
            Text("Selected: ${gb(total)}. Free on this phone: ${gb(free)}." +
                if (total > free) " Not enough space for all of it." else "", style = MaterialTheme.typography.bodySmall)
            Text("Use Wi-Fi. The downloads run side by side and carry on if you leave the app.",
                style = MaterialTheme.typography.bodySmall)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    // only the small chat model chosen: chat uses it (the "Lowest" setting) instead of waiting for Ling
                    val chat = picked.filter { it in ModelStore.SETUP_CHAT }
                    if (chat.size == 1 && chat[0] == ModelStore.SMALLCHAT) LocalLing.setExpertSlots(ctx, LocalLing.SMALL)
                    if (chat.size == 1 && chat[0] == ModelStore.LING && LocalLing.small(ctx)) LocalLing.setExpertSlots(ctx, 0)
                    for (st in picked) {
                        val m = manifests[st.id] ?: continue
                        Installer.start(ctx, st, m, m.variants.first().id) { Engine.release() }
                    }
                    started = true
                }, enabled = picked.isNotEmpty() && total <= free) {
                    Text(if (picked.size == 1) "Download 1 model" else "Download ${picked.size} models")
                }
                OutlinedButton(onClick = onDone) { Text(if (firstRun) "Skip" else "Back") }
            }
        } else {
            Text(if (Installer.running) "Downloading. You can start using the app; each model appears as it finishes."
                else "All done.", style = MaterialTheme.typography.bodySmall)
            Button(onClick = onDone) { Text(if (Installer.running) "Continue to the app" else if (firstRun) "Start" else "Done") }
        }
    }
}
