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
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.sp
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.setValue
import androidx.activity.compose.BackHandler
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

internal fun stateText(j: Job, progress: Pair<String, String>): String = when (j.state) {
    JobState.WAITING -> "Waiting"
    JobState.DRAWING -> "Drawing" + (if (progress.first == j.id && progress.second.isNotEmpty()) ": ${progress.second}" else "")
    JobState.DONE -> "Done in %.0f s".format(j.ms / 1000.0)
    JobState.FAILED -> j.error?.trim()?.takeIf { it.isNotEmpty() }?.let { "Failed · $it" } ?: "Failed"
    JobState.CANCELLED -> "Cancelled"
}

internal fun modelName(j: Job) = when (j.model) {
    "qwen" -> if (j.qwenDrawing == "noctq") "Noct Q" else "Qwen-Image"
    "fibo" -> "Fibo edit"
    "moss" -> "MOSS"
    "wan" -> "Wan clip"
    "upscale" -> "Upscale"
    "klein" -> "Klein"
    "krea" -> "Krea"
    else -> "Anima"
}

/** How many jobs a section shows at first, and how many more each "Show more" adds: hundreds at once made the page slow. */
internal const val QUEUE_PAGE = 30

/** The "Show more" button's text, or null when every job is already shown. */
internal fun moreLabel(total: Int, shown: Int): String? =
    (total - shown).takeIf { it > 0 }?.let { "Show ${minOf(it, QUEUE_PAGE)} more" }

/** A drawing picture's one line in chat: the progress alone ("Step 4 of 8"), the spinner beside it says it is drawing. */
internal fun drawingLine(progress: String): String =
    progress.trim().replaceFirstChar { it.uppercase() }.ifEmpty { "Drawing…" }

/** Every job, newest first: cancel what is waiting, retry what failed, open what is done. */
@Composable
fun QueueScreen(onOpen: (Gallery.Entry) -> Unit, onBack: () -> Unit) {
    val ctx = LocalContext.current
    // Back returns to the page the queue was opened from instead of leaving the app
    BackHandler(onBack = onBack)
    val tick by Queue.store.changed.collectAsState()
    val progress by QueueService.progress.collectAsState()
    val jobs = remember(tick) { Queue.store.all().reversed() }
    val thumbs = remember { mutableStateMapOf<String, Bitmap>() }
    var shownWaiting by remember { mutableIntStateOf(QUEUE_PAGE) }
    var shownFinished by remember { mutableIntStateOf(QUEUE_PAGE) }
    LaunchedEffect(tick, shownFinished) {
        withContext(Dispatchers.IO) {
            // only the finished jobs on screen: hundreds of thumbnails at once made the page slow
            val onScreen = jobs.filter { it.state == JobState.DONE || it.state == JobState.FAILED || it.state == JobState.CANCELLED }.take(shownFinished)
            for (j in onScreen) if (j.state == JobState.DONE && j.resultUri != null && j.id !in thumbs) {
                val e = Gallery.Entry(android.net.Uri.parse(j.resultUri), j.prompt, j.seed, j.steps, j.ms, j.width, 0, j.height)
                Gallery.thumbnail(ctx, e, 192)?.let { thumbs[j.id] = it }
            }
        }
    }
    val p = LocalPaper.current
    val drawing = jobs.filter { it.state == JobState.DRAWING }
    val waiting = jobs.filter { it.state == JobState.WAITING }.reversed()   // next to draw at the top
    val finished = jobs.filter { it.state == JobState.DONE || it.state == JobState.FAILED || it.state == JobState.CANCELLED }

    @Composable fun Thumb(j: Job) {
        val shape = RoundedCornerShape(10.dp)
        val bad = j.state == JobState.FAILED || j.state == JobState.CANCELLED
        Box(Modifier.size(44.dp).clip(shape).background(if (bad) p.badSoft else p.soft), contentAlignment = Alignment.Center) {
            thumbs[j.id]?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
                ?: Text(if (bad) "!" else modelName(j).take(1), color = if (bad) p.bad else p.softText, fontWeight = FontWeight.Bold)
        }
    }
    @Composable fun JobRow(j: Job, last: Boolean) = PaperRow(divider = !last) {
        if (j.state != JobState.WAITING) Thumb(j)
        Column(Modifier.weight(1f)) {
            Text(j.prompt.lineSequence().first(), fontSize = 14.sp, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text("${modelName(j)} · ${stateText(j, progress)}", fontSize = 12.sp, maxLines = 2, overflow = TextOverflow.Ellipsis,
                color = if (j.state == JobState.FAILED) p.bad else p.muted)
        }
        when (j.state) {
            JobState.WAITING -> TextButton(onClick = { Queue.store.update(j.id) { it.copy(state = JobState.CANCELLED) } }) {
                Text("Cancel", color = p.muted) }
            JobState.FAILED, JobState.CANCELLED -> Pill("Retry") {
                Queue.store.update(j.id) { it.copy(state = JobState.WAITING, error = null, tries = 0) }
                QueueService.kick(ctx)
            }
            JobState.DONE -> Pill("Open") {
                j.resultUri?.let { u -> onOpen(Gallery.Entry(android.net.Uri.parse(u), j.prompt, j.seed, j.steps, j.ms, j.width, 0, j.height)) }
            }
            JobState.DRAWING -> {}
        }
    }
    @Composable fun DrawingCard(j: Job, big: Boolean) = PaperCard(Modifier.fillMaxWidth(), padding = PaddingValues(14.dp)) {
        val step = if (progress.first == j.id) Regex("""step (\d+) of (\d+)""").find(progress.second) else null
        if (big) Box(Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(14.dp)).background(p.soft),
            contentAlignment = Alignment.Center) { Icon(AppIcons.Brush, null, Modifier.size(32.dp), tint = p.softText) }
        if (big) Spacer(Modifier.height(12.dp))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            if (!big) Box(Modifier.size(56.dp).clip(RoundedCornerShape(12.dp)).background(p.soft), contentAlignment = Alignment.Center) {
                Icon(AppIcons.Brush, null, Modifier.size(22.dp), tint = p.softText) }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(j.prompt.lineSequence().first(), fontSize = 14.sp, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(modelName(j) + " · " + (if (progress.first == j.id && progress.second.isNotEmpty()) progress.second else "starting"),
                    fontSize = 12.sp, color = p.muted)
                val m = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp))
                if (step != null) LinearProgressIndicator(progress = { step.groupValues[1].toFloat() / step.groupValues[2].toFloat() }, m,
                    color = p.badge, trackColor = p.divider, drawStopIndicator = {})
                else LinearProgressIndicator(m, color = p.badge, trackColor = p.divider)
            }
        }
    }
    @Composable fun Section(title: String, list: List<Job>, shown: Int, onMore: () -> Unit) {
        if (list.isEmpty()) return
        SectionLabel(title)
        val rows = list.take(shown)
        PaperCard(Modifier.fillMaxWidth(), padding = PaddingValues(horizontal = 14.dp)) {
            rows.forEachIndexed { i, j -> JobRow(j, last = i == rows.lastIndex) }
        }
        moreLabel(list.size, shown)?.let { Pill(it, onClick = onMore) }
    }

    Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().height(64.dp).padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) { Icon(AppIcons.Back, "Back") }
            Text("Queue", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f).padding(start = 4.dp))
            TextButton(onClick = {
                Queue.store.remove(Queue.store.all().filter { it.state == JobState.DONE || it.state == JobState.FAILED || it.state == JobState.CANCELLED }.map { it.id }.toSet())
            }, enabled = finished.isNotEmpty()) { Text("Clear finished", color = if (finished.isNotEmpty()) p.link else p.muted) }
        }
        if (jobs.isEmpty()) Text("Nothing in the queue.", Modifier.padding(16.dp), color = p.muted)
        val left: @Composable ColumnScope.(Boolean) -> Unit = { big ->
            if (drawing.isNotEmpty()) { SectionLabel("Drawing now"); drawing.forEach { DrawingCard(it, big) } }
            Section("Waiting · ${waiting.size}", waiting, shownWaiting) { shownWaiting += QUEUE_PAGE }
        }
        if (isWide()) Row(Modifier.fillMaxSize().padding(horizontal = 16.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                left(true); Spacer(Modifier.height(16.dp)) }
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Section("Finished", finished, shownFinished) { shownFinished += QUEUE_PAGE }; Spacer(Modifier.height(16.dp)) }
        } else Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            left(false); Section("Finished", finished, shownFinished) { shownFinished += QUEUE_PAGE }; Spacer(Modifier.height(16.dp))
        }
    }
}
