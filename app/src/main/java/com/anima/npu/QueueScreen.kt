package com.anima.npu

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

/** Adds what the Create screen shows to the queue (the guide or edit picture saved beside the queue file). */
fun enqueueFromCreate(ctx: Context, s: CreateState): Job {
    val id = UUID.randomUUID().toString().take(12)
    val pic = if (s.model == "anima") s.guide else s.edit
    val src = pic?.let { b ->
        val name = "$id-src.png"
        File(Queue.dir(ctx), name).outputStream().use { b.compress(Bitmap.CompressFormat.PNG, 100, it) }
        name
    }
    val j = Job(id, System.currentTimeMillis(), s.model, s.qwenDrawing, s.prompt, s.seed, s.steps, s.tier, s.shape,
        s.guideStrength, s.guideEnd, src, "create")
    Queue.store.add(j)
    QueueService.kick(ctx)
    return j
}

private fun stateText(j: Job, progress: Pair<String, String>): String = when (j.state) {
    JobState.WAITING -> "Waiting"
    JobState.DRAWING -> "Drawing" + (if (progress.first == j.id && progress.second.isNotEmpty()) ": ${progress.second}" else "")
    JobState.DONE -> "Done in %.0f s".format(j.ms / 1000.0)
    JobState.FAILED -> "Failed: ${j.error ?: ""}"
    JobState.CANCELLED -> "Cancelled"
}

private fun modelName(j: Job) = when (j.model) {
    "qwen" -> if (j.qwenDrawing == "noctq") "Noct Q" else "Qwen-Image"
    "klein" -> "Klein"
    else -> "Anima"
}

/** Every job, newest first: cancel what is waiting, retry what failed, open what is done. */
@Composable
fun QueueScreen(onOpen: (Gallery.Entry) -> Unit, onBack: () -> Unit) {
    val ctx = LocalContext.current
    val tick by Queue.store.changed.collectAsState()
    val progress by QueueService.progress.collectAsState()
    val jobs = remember(tick) { Queue.store.all().reversed() }
    val thumbs = remember { mutableStateMapOf<String, Bitmap>() }
    LaunchedEffect(tick) {
        withContext(Dispatchers.IO) {
            for (j in jobs) if (j.state == JobState.DONE && j.resultUri != null && j.id !in thumbs) {
                val e = Gallery.Entry(android.net.Uri.parse(j.resultUri), j.prompt, j.seed, j.steps, j.ms, j.width, 0, j.height)
                Gallery.thumbnail(ctx, e, 192)?.let { thumbs[j.id] = it }
            }
        }
    }
    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onBack) { Text("‹ Back") }
            Text("Queue", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
            TextButton(onClick = {
                Queue.store.remove(Queue.store.all().filter { it.state == JobState.DONE || it.state == JobState.FAILED || it.state == JobState.CANCELLED }.map { it.id }.toSet())
            }) { Text("Clear finished") }
        }
        if (jobs.isEmpty()) Text("Nothing in the queue.", Modifier.padding(top = 16.dp))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(jobs, key = { it.id }) { j ->
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Box(Modifier.size(64.dp)) {
                        thumbs[j.id]?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
                            ?: Text(modelName(j), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.Center))
                    }
                    Column(Modifier.weight(1f)) {
                        Text(j.prompt.lineSequence().first().take(80), style = MaterialTheme.typography.bodySmall, maxLines = 2)
                        Text("${modelName(j)} · ${stateText(j, progress)}", style = MaterialTheme.typography.labelSmall, maxLines = 2)
                    }
                    when (j.state) {
                        JobState.WAITING -> TextButton(onClick = { Queue.store.update(j.id) { it.copy(state = JobState.CANCELLED) } }) { Text("Cancel") }
                        JobState.FAILED, JobState.CANCELLED -> TextButton(onClick = {
                            Queue.store.update(j.id) { it.copy(state = JobState.WAITING, error = null, tries = 0) }
                            QueueService.kick(ctx)
                        }) { Text("Retry") }
                        JobState.DONE -> TextButton(onClick = {
                            j.resultUri?.let { u -> onOpen(Gallery.Entry(android.net.Uri.parse(u), j.prompt, j.seed, j.steps, j.ms, j.width, 0, j.height)) }
                        }) { Text("Open") }
                        JobState.DRAWING -> {}
                    }
                }
            }
        }
    }
}
