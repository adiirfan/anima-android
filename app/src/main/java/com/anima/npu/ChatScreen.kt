package com.anima.npu

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.DateFormat
import java.util.Date

/** The Chat tab: talk to Ling, which can add pictures to the queue; finished pictures appear in the conversation. */
@Composable
fun ChatScreen(chat: ChatController, lingInstalled: Boolean, onDownload: () -> Unit, onOpen: (Gallery.Entry) -> Unit) {
    val ctx = LocalContext.current
    if (!lingInstalled && !LocalLing.ready(ctx)) {
        Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Chat needs Ling", style = MaterialTheme.typography.titleMedium)
            Text("Ling-3.0-tiny runs on the phone's AI chip and can add pictures to the queue for you. " +
                "It is a 4.5 GB download.")
            Button(onClick = onDownload) { Text("Download Ling") }
        }
        return
    }
    val prefs = remember { ctx.getSharedPreferences("anima", android.content.Context.MODE_PRIVATE) }
    var convId by remember { mutableStateOf(prefs.getString("chatConv", null)) }
    var showList by remember { mutableStateOf(false) }
    var newThinking by remember { mutableStateOf(false) }
    val tick by chat.changed.collectAsState()
    val turn by chat.state.collectAsState()
    val conv = remember(tick, convId) { convId?.let { chat.store().get(it) } }
    fun open(id: String?) { convId = id; prefs.edit().putString("chatConv", id).apply() }

    if (showList) {
        ChatList(chat, current = convId, onOpen = { open(it); showList = false }, onBack = { showList = false })
        return
    }
    var text by remember { mutableStateOf("") }
    var attached by remember { mutableStateOf<Bitmap?>(null) }
    var renaming by remember { mutableStateOf(false) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        if (uri != null) attached = runCatching {
            ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
        }.getOrNull()
    }
    val busy = turn.phase != "idle"
    val thinking = conv?.thinking ?: newThinking
    Column(Modifier.fillMaxSize().imePadding()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            if (renaming && conv != null) {
                var t by remember { mutableStateOf(conv.title) }
                OutlinedTextField(t, { t = it }, Modifier.weight(1f), singleLine = true)
                TextButton(onClick = { chat.store().rename(conv.id, t.ifBlank { conv.title }); chat.changed.value++; renaming = false }) { Text("OK") }
            } else {
                Text(conv?.title ?: "New chat", style = MaterialTheme.typography.titleMedium, maxLines = 1,
                    modifier = Modifier.weight(1f).clickable(enabled = conv != null) { renaming = true })
                TextButton(onClick = { showList = true }) { Text("Chats") }
                TextButton(onClick = { open(null) }, enabled = !busy) { Text("New") }
            }
        }
        Row(Modifier.padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Think first", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(end = 8.dp))
            Switch(thinking, { on -> if (conv != null) chat.setThinking(conv.id, on) else newThinking = on }, enabled = !busy)
            Spacer(Modifier.weight(1f))
            val lingState = when (turn.phase) {
                "waiting" -> "Waiting for the current picture…"
                "loading" -> "Loading Ling…"
                "thinking" -> "Thinking…"
                "tool" -> "Adding pictures…"
                else -> ""
            }
            Text(lingState, style = MaterialTheme.typography.bodySmall)
        }
        val list = rememberLazyListState()
        val msgs = conv?.msgs ?: emptyList()
        val live = busy && turn.convId == convId && (turn.partialText.isNotEmpty() || turn.partialThinking.isNotEmpty())
        // newest at the bottom, and the list anchored there (reversed), so a chat always opens at its end and
        // follows the reply as it streams in
        LaunchedEffect(msgs.size, live) { list.scrollToItem(0) }
        LazyColumn(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp), state = list, reverseLayout = true,
            verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.Bottom)) {
            if (live) item { AssistantBubble(turn.partialText, turn.partialThinking) }
            items(msgs.asReversed(), key = { it.id }) { m -> MessageRow(m, chat.store().dir(), onOpen) }
            if (msgs.isEmpty() && !live) item {
                Text("Ask for a picture, for example \"draw an anime girl in the rain\" or \"a photo of a red bicycle by a canal\".",
                    style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(8.dp))
            }
        }
        attached?.let { b ->
            Row(Modifier.padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                Image(b.asImageBitmap(), null, Modifier.size(56.dp).clip(RoundedCornerShape(6.dp)), contentScale = ContentScale.Crop)
                TextButton(onClick = { attached = null }) { Text("Remove picture") }
            }
        }
        Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            OutlinedButton(onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                enabled = !busy, contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp)) { Text("＋") }
            OutlinedTextField(text, { text = it }, Modifier.weight(1f), placeholder = { Text("Message") }, maxLines = 4)
            if (busy) Button(onClick = { chat.stop() }) { Text("Stop") }
            else Button(onClick = {
                val id = chat.send(convId, text.trim(), attached, thinking)
                open(id); text = ""; attached = null
            }, enabled = text.isNotBlank()) { Text("Send") }
        }
    }
}

@Composable
private fun MessageRow(m: Msg, dir: File, onOpen: (Gallery.Entry) -> Unit) {
    when (m) {
        is Msg.User -> Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.CenterEnd) {
            Column(Modifier.widthIn(max = 300.dp).clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.primaryContainer).padding(10.dp)) {
                m.imageFile?.let { f ->
                    val b by produceState<Bitmap?>(null, f) {
                        value = withContext(Dispatchers.IO) { decodeSmall(File(dir, f), 256) }
                    }
                    b?.let { Image(it.asImageBitmap(), null, Modifier.size(120.dp).clip(RoundedCornerShape(6.dp)), contentScale = ContentScale.Crop) }
                }
                if (m.text.isNotBlank()) Text(m.text, color = MaterialTheme.colorScheme.onPrimaryContainer)
            }
        }
        is Msg.Assistant -> AssistantBubble(m.text, m.thinking)
        is Msg.ToolCall -> {
            val a = runCatching { org.json.JSONObject(m.argsJson) }.getOrNull()
            val model = when (a?.optString("model")) { "qwen" -> "Qwen"; "anima" -> "Anima"; else -> a?.optString("model") ?: "?" }
            Text("Queued: $model · ${a?.optString("prompt")?.take(60) ?: m.argsJson.take(60)}",
                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.outline)
        }
        is Msg.ToolResult -> {
            val err = runCatching { org.json.JSONObject(m.resultJson).optString("error") }.getOrNull()
            if (!err.isNullOrEmpty()) Text("Not queued: $err", style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.error)
        }
        is Msg.Picture -> PictureCard(m.jobId, onOpen)
    }
}

@Composable
private fun AssistantBubble(text: String, thinking: String) {
    var open by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(end = 40.dp)) {
        if (thinking.isNotBlank()) {
            Text(if (open) "Thought ▾" else "Thought for a moment ▸", style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline, modifier = Modifier.clickable { open = !open })
            if (open) Text(thinking, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.outline)
        }
        if (text.isNotBlank()) Text(text)
    }
}

/** A queued picture: progress while it waits or draws, the picture once done (tap to open it). */
@Composable
private fun PictureCard(jobId: String, onOpen: (Gallery.Entry) -> Unit) {
    val ctx = LocalContext.current
    val tick by Queue.store.changed.collectAsState()
    val progress by QueueService.progress.collectAsState()
    val j = remember(tick) { Queue.store.get(jobId) }
    val entry = j?.takeIf { it.state == JobState.DONE && it.resultUri != null }?.let {
        Gallery.Entry(Uri.parse(it.resultUri), it.prompt, it.seed, it.steps, it.ms, it.width, 0, it.height)
    }
    val bmp by produceState<Bitmap?>(null, entry) {
        value = entry?.let { e -> withContext(Dispatchers.IO) { Gallery.thumbnail(ctx, e, 512) } }
    }
    Box(Modifier.widthIn(max = 280.dp).fillMaxWidth().clip(RoundedCornerShape(10.dp))
        .background(MaterialTheme.colorScheme.surfaceVariant)) {
        val b = bmp
        if (b != null && entry != null) {
            Image(b.asImageBitmap(), null, Modifier.fillMaxWidth().aspectRatio(b.width.toFloat() / b.height)
                .clickable { onOpen(entry) }, contentScale = ContentScale.Fit)
        } else {
            val what = when (j?.state) {
                null -> "This picture is no longer in the queue"
                JobState.WAITING -> "Waiting in the queue…"
                JobState.DRAWING -> "Drawing" + (if (progress.first == jobId && progress.second.isNotEmpty()) ": ${progress.second}" else "…")
                JobState.FAILED -> "Failed: ${j.error ?: ""}"
                JobState.CANCELLED -> "Cancelled"
                JobState.DONE -> "Loading…"
            }
            Column {
                Text(what, Modifier.padding(16.dp), style = MaterialTheme.typography.bodySmall)
                if (j?.state == JobState.FAILED || j?.state == JobState.CANCELLED) TextButton(onClick = {
                    Queue.store.update(j.id) { it.copy(state = JobState.WAITING, error = null, tries = 0) }
                    QueueService.kick(ctx)
                }, modifier = Modifier.padding(start = 8.dp)) { Text("Retry") }
            }
        }
    }
}

@Composable
private fun ChatList(chat: ChatController, current: String?, onOpen: (String?) -> Unit, onBack: () -> Unit) {
    val tick by chat.changed.collectAsState()
    val all = remember(tick) { chat.store().list() }
    val fmt = remember { DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT) }
    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onBack) { Text("‹ Back") }
            Text("Chats", style = MaterialTheme.typography.titleMedium)
        }
        if (all.isEmpty()) Text("No chats yet.", Modifier.padding(top = 16.dp))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            items(all, key = { it.id }) { c ->
                Row(Modifier.fillMaxWidth().clickable { onOpen(c.id) }.padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(c.title, maxLines = 1)
                        Text(fmt.format(Date(c.updated)), style = MaterialTheme.typography.labelSmall)
                    }
                    TextButton(onClick = {
                        chat.store().delete(c.id); chat.changed.value++
                        if (c.id == current) onOpen(null)
                    }, enabled = !(chat.busy() && chat.state.value.convId == c.id)) { Text("Delete") }
                }
            }
        }
    }
}

private fun decodeSmall(f: File, px: Int): Bitmap? {
    val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(f.absolutePath, o)
    var s = 1
    while (o.outWidth / (s * 2) >= px && o.outHeight / (s * 2) >= px) s *= 2
    return BitmapFactory.decodeFile(f.absolutePath, BitmapFactory.Options().apply { inSampleSize = s })
}
