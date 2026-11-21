package com.anima.npu

import android.os.StatFs
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private fun gb(bytes: Long) = "%.1f GB".format(bytes / 1e9)

/**
 * The models page (Settings): every model with its builds — the AI-chip one when this phone's chip has it, and the
 * graphics-chip one when the model has it. Each build can be downloaded, deleted, downloaded again or updated on
 * its own, and a model with both picks which one draws. Updates fetch only the files that changed.
 * [onChanged]: something was installed, deleted or switched (the app re-reads what is installed).
 */
@Composable
fun ModelsScreen(onChanged: () -> Unit, onBack: () -> Unit) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    BackHandler(onBack = onBack)
    // every model is listed: one with no build for this phone says so instead of quietly missing
    val models = ModelStore.SETUP_PICTURE + ModelStore.SETUP_CHAT
    val builds = models.flatMap { listOfNotNull(it.takeIf { m -> m.aiChip }, it.gpu?.takeIf { g -> g.available }) }
    val manifests = remember { mutableStateMapOf<String, ModelStore.Manifest>() }
    val fresh = remember { mutableStateMapOf<String, Boolean>() }   // the list came from Hugging Face just now (updates only then)
    var tick by remember { mutableIntStateOf(0) }       // bumped after a delete or a switch, to read the folders again
    var checking by remember { mutableStateOf(false) }
    var checked by remember { mutableStateOf("") }
    var confirm by remember { mutableStateOf<Pair<ModelStore, Boolean>?>(null) }   // (build, true = download again)
    val qTick by Queue.store.changed.collectAsState()

    // the lists from Hugging Face (a few kilobytes each, all at once); the cached ones stand in when offline
    suspend fun check() {
        checking = true
        val failed = coroutineScope {
            builds.map { b ->
                async(Dispatchers.IO) {
                    val m = runCatching { b.fetchManifest(ctx) }.getOrNull()
                    (m ?: b.cachedManifest(ctx))?.let { manifests[b.id] = it }
                    m?.let { b.noteVersion(ctx, it) }
                    fresh[b.id] = m != null
                    b.title.takeIf { m == null }
                }
            }.awaitAll().filterNotNull().distinct()
        }
        checking = false
        tick++
        checked = if (failed.isNotEmpty()) "Could not get the file list of ${failed.joinToString(", ")} from Hugging Face, " +
            "so updates for it are not shown." else "All up to date as of now, apart from any update shown below."
    }
    LaunchedEffect(Unit) {
        builds.forEach { b -> b.cachedManifest(ctx)?.let { manifests[b.id] = it } }
        check()
    }
    // a download that finished while this page is open
    LaunchedEffect(Installer.tasks.values.count { !it.running }) { tick++; onChanged() }

    /** The model is drawing (or Ling answering) right now: its files must not go away under it. */
    fun inUse(m: ModelStore) = when (m.baseId) {
        "ling", "smallchat" -> Chat.busy()
        else -> Queue.store.all().any { it.state == JobState.DRAWING && it.model == m.baseId }
    }

    fun start(b: ModelStore, again: Boolean) {
        val m = manifests[b.id] ?: return
        scope.launch {
            if (again) withContext(Dispatchers.IO) { free(b); b.delete(ctx) }
            Installer.start(ctx, b, m, b.installed(ctx) ?: m.variants.first().id) { Engine.release() }
            tick++
        }
    }

    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        TextButton(onClick = onBack) { Text("‹ Back") }
        Text("Model Management", style = MaterialTheme.typography.headlineSmall)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = { scope.launch { check() } }, enabled = !checking) {
                Text(if (checking) "Checking…" else "Check for updates")
            }
            val free = remember(tick) { runCatching { StatFs(ctx.getExternalFilesDir(null)!!.path).availableBytes }.getOrDefault(0L) }
            Text("Free: ${gb(free)}", style = MaterialTheme.typography.bodySmall)
        }
        if (checked.isNotEmpty()) Text(checked, style = MaterialTheme.typography.bodySmall)

        for (model in models) {
            // read again after deletes, switches and finished downloads
            val gpuOn = remember(tick) { model.runsOnGpu(ctx) }
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(model.title, style = MaterialTheme.typography.titleMedium)
                    val g = model.gpu?.takeIf { it.available }
                    // every model shows the choice; a side with no build yet is greyed out and says so
                    run {
                        Text(if (model in ModelStore.SETUP_CHAT) "Runs on" else "Draws on", style = MaterialTheme.typography.labelMedium)
                        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                            listOf(false to "AI chip", true to "Graphics chip").forEachIndexed { i, (on, label) ->
                                val exists = if (on) g != null || model.sameFilesOnGpu else model.aiChip
                                SegmentedButton(selected = exists && gpuOn == on, enabled = exists, onClick = {
                                    if (gpuOn != on) { model.setRunsOnGpu(ctx, on); tick++; onChanged() }
                                }, shape = SegmentedButtonDefaults.itemShape(i, 2)) { Text(label) }
                            }
                        }
                        if (!model.aiChip) Text("AI chip: no build for this phone yet.", style = MaterialTheme.typography.bodySmall)
                        if (g == null && !model.sameFilesOnGpu) Text("Graphics chip: not built yet.", style = MaterialTheme.typography.bodySmall)
                    }
                    for (b in listOfNotNull(model.takeIf { it.aiChip }, g)) {
                        if (model.aiChip && g != null) HorizontalDivider()
                        BuildRow(b, manifests[b.id], fresh[b.id] == true, tick + qTick.hashCode(), inUse(b), onStart = { again -> start(b, again) },
                            onDelete = { confirm = b to false }, onRedownload = { confirm = b to true })
                    }
                }
            }
        }
        Text("Licences: " + models.joinToString(" ") { it.licence }, style = MaterialTheme.typography.bodySmall)
    }

    confirm?.let { (b, again) ->
        val where = if (b.onGraphicsChip) " (graphics chip)" else if (ModelStore.byId(b.baseId).gpu?.available == true) " (AI chip)" else ""
        AlertDialog(onDismissRequest = { confirm = null },
            title = { Text((if (again) "Download ${b.title} again?" else "Delete ${b.title}?") + where) },
            text = { Text(if (again) "Its files are deleted, then downloaded fresh from Hugging Face. Use this if the model " +
                "draws wrongly or a file got damaged." else "Its files are removed from the phone. You can download it " +
                "again later.") },
            confirmButton = { TextButton(onClick = {
                confirm = null
                if (again) start(b, true)
                else scope.launch { withContext(Dispatchers.IO) { free(b); b.delete(ctx) }; tick++; onChanged() }
            }) { Text(if (again) "Download again" else "Delete") } },
            dismissButton = { TextButton(onClick = { confirm = null }) { Text("Cancel") } })
    }
}

/** Lets go of whatever has the build's files open: the AI chip's pieces, or the chat model. */
private fun free(b: ModelStore) {
    if (b.baseId == "ling" || b.baseId == "smallchat") LocalLing.stop()
    Engine.release()
    Engines.invalidate(b.baseId)
}

/** One build of a model: what is on the phone, what an update would fetch, and its buttons. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun BuildRow(
    b: ModelStore, m: ModelStore.Manifest?, fresh: Boolean, tick: Int, inUse: Boolean,
    onStart: (Boolean) -> Unit, onDelete: () -> Unit, onRedownload: () -> Unit,
) {
    val ctx = LocalContext.current
    val uri = LocalUriHandler.current
    val task = Installer.task(b.id)
    val busy = task?.running == true
    val variant = remember(tick, busy) { b.installed(ctx) }
    val version = remember(tick, busy) { b.installedVersion(ctx) }
    // only against a list fetched just now: a cached one can be older than the files on the phone
    val changed = remember(tick, busy, m, fresh) { if (m != null && fresh && variant != null) b.outdated(ctx, m) else emptyList() }
    val label = if (b.onGraphicsChip) "Graphics chip" else if (b.gpu?.available == true) "AI chip" else null
    val size = m?.let { it.bytes(variant ?: it.variants.first().id) }

    Text(buildString {
        label?.let { append(it).append(": ") }
        append(if (variant != null) "Downloaded, version $version" else "Not downloaded")
        size?.let { append(" · ").append(gb(it)) }
    }, style = MaterialTheme.typography.bodyMedium)
    if (variant != null && m != null && changed.isNotEmpty()) {
        Text((if (m.version > version) "Update to version ${m.version}" else "Update available") + (m.notes.trimEnd('.', ' ').takeIf { it.isNotBlank() }?.let { ": $it" } ?: "") +
            ". Downloads ${gb(changed.sumOf { it.size })} (${changed.size} of ${(m.shared + m.variant(variant).files).size} files).",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
    }
    if (inUse) Text("In use right now. Delete and download again once it has finished.", style = MaterialTheme.typography.bodySmall)
    task?.let { t ->
        if (t.running || t.failed) {
            if (t.running) LinearProgressIndicator(progress = { t.fraction }, modifier = Modifier.fillMaxWidth())
            Text(t.status, style = MaterialTheme.typography.bodySmall)
        }
    }
    // wraps onto a second line when Update, Download again and Delete do not fit side by side
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        when {
            busy -> OutlinedButton(onClick = { Installer.cancel(b.id) }) { Text("Cancel") }
            variant == null -> Button(onClick = { onStart(false) }, enabled = m != null) { Text("Download") }
            else -> {
                if (changed.isNotEmpty()) Button(onClick = { onStart(false) }, enabled = !inUse) { Text("Update") }
                OutlinedButton(onClick = onRedownload, enabled = m != null && !inUse) { Text("Download again") }
                OutlinedButton(onClick = onDelete, enabled = !inUse) { Text("Delete") }
            }
        }
        b.pageUrl?.let { url -> OutlinedButton(onClick = { uri.openUri(url) }) { Text("Model page") } }
    }
}
