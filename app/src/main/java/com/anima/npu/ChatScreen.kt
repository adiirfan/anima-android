package com.anima.npu

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.DateFormat
import java.util.Date

/**
 * The Chat tab, laid out like sysmon's phone chat: a slim bar (menu, the chat's name with the model under it, the
 * list of chats, a menu), your messages in grey bubbles on the right, replies as plain text across the page, tool
 * steps as one quiet line, and one rounded composer holding the text, attach, think, the model and send.
 */
@Composable
fun ChatScreen(chat: ChatController, isInstalled: (String) -> Boolean, onDownload: () -> Unit,
               onOpen: (Gallery.Entry) -> Unit, onMenu: () -> Unit) {
    val ctx = LocalContext.current
    // which chat model (Settings or the pill in the composer); everything that depends on it reads this
    var slots by remember { mutableIntStateOf(LocalLing.expertSlots(ctx)) }
    val ready = remember(slots) { LocalLing.ready(ctx) } || isInstalled(LocalLing.store(ctx).id)
    if (!ready) {
        Column(Modifier.fillMaxSize()) {
            ChatBar(onMenu, "Chat", LocalLing.name(ctx))
            Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Chat needs ${LocalLing.name(ctx)}", style = MaterialTheme.typography.titleMedium)
                Text(if (LocalLing.small(ctx)) "Qwen3.5 0.8B is the small chat model (the Lowest setting). It runs on " +
                        "the phone's AI chip and can add pictures to the queue for you. It is a 0.8 GB download."
                    else "Ling-3.0-tiny runs on the phone's AI chip and can add pictures to the queue for you. " +
                        "It is a 4.5 GB download. (Settings has a smaller model for phones short of memory.)")
                Button(onClick = onDownload) { Text("Download ${LocalLing.name(ctx)}") }
            }
        }
        return
    }
    val prefs = remember { ctx.getSharedPreferences("anima", Context.MODE_PRIVATE) }
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
    val mine = busy && turn.convId == convId
    val thinking = conv?.thinking ?: newThinking
    val status = when (turn.phase) {
        "waiting" -> "Waiting for the current picture…"
        "loading" -> "Loading ${LocalLing.name(ctx)}…"
        "preparing" -> "Preparing Ling for low memory (first time, about a minute)…"
        "thinking" -> "Thinking…"
        "tool" -> "Adding pictures…"
        "writing" -> "Writing…"
        else -> ""
    }

    Column(Modifier.fillMaxSize().imePadding()) {
        ChatBar(onMenu, conv?.title ?: "New chat", if (mine) status else modelLabel(ctx, slots), buttons = 2) {
            IconButton(onClick = { showList = true }) { Icon(AppIcons.Chats, "Chats") }
            var menu by remember { mutableStateOf(false) }
            Box {
                IconButton(onClick = { menu = true }) { Icon(AppIcons.More, "More") }
                DropdownMenu(menu, { menu = false }) {
                    DropdownMenuItem({ Text("New chat") }, { menu = false; open(null) },
                        leadingIcon = { Icon(AppIcons.Add, null) }, enabled = !busy)
                    DropdownMenuItem({ Text("Rename") }, { menu = false; renaming = true },
                        leadingIcon = { Icon(AppIcons.Edit, null) }, enabled = conv != null)
                    DropdownMenuItem({ Text("Delete chat") }, {
                        menu = false
                        conv?.let { chat.store().delete(it.id); chat.changed.value++; open(null) }
                    }, leadingIcon = { Icon(AppIcons.Delete, null) }, enabled = conv != null && !mine)
                }
            }
        }
        if (renaming && conv != null) RenameDialog(conv.title, onDone = { t ->
            if (t != null) { chat.store().rename(conv.id, t.ifBlank { conv.title }); chat.changed.value++ }
            renaming = false
        })

        val list = rememberLazyListState()
        val msgs = conv?.msgs ?: emptyList()
        val live = mine && (turn.partialText.isNotEmpty() || turn.partialThinking.isNotEmpty())
        // newest at the bottom, and the list anchored there (reversed), so a chat always opens at its end and
        // follows the reply as it streams in
        LaunchedEffect(msgs.size, live) { list.scrollToItem(0) }
        if (msgs.isEmpty() && !mine) {
            Column(Modifier.weight(1f).fillMaxWidth().padding(32.dp), verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Ready when you are.", fontSize = 22.sp, fontWeight = FontWeight.Medium)
                Spacer(Modifier.height(10.dp))
                Text("Ask for a picture, like \"draw an anime girl in the rain\", then say \"edit it\" to change it.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else LazyColumn(Modifier.weight(1f).fillMaxWidth(), state = list, reverseLayout = true,
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp, Alignment.Bottom)) {
            if (live) item { Reply(turn.partialText, turn.partialThinking, actions = false) }
            else if (mine) item { StatusLine(status) }
            items(msgs.asReversed(), key = { it.id }) { m -> MessageRow(m, chat.store().dir(), onOpen) }
        }

        // the picture going with the next message
        attached?.let { b ->
            Box(Modifier.padding(start = 20.dp, top = 6.dp, bottom = 4.dp)) {
                Image(b.asImageBitmap(), null, Modifier.size(56.dp).clip(RoundedCornerShape(10.dp)), contentScale = ContentScale.Crop)
                Surface(Modifier.align(Alignment.TopEnd).padding(2.dp).size(20.dp).clickable { attached = null },
                    shape = CircleShape, color = MaterialTheme.colorScheme.surfaceContainerHighest) {
                    Icon(AppIcons.Close, "Remove picture", Modifier.padding(3.dp))
                }
            }
        }

        // the composer
        Column(Modifier.padding(start = 12.dp, end = 12.dp, top = 4.dp, bottom = 12.dp).fillMaxWidth()
            .clip(RoundedCornerShape(26.dp)).background(MaterialTheme.colorScheme.surfaceContainerHigh)
            .padding(start = 16.dp, end = 10.dp, top = 12.dp, bottom = 8.dp)) {
            val scheme = MaterialTheme.colorScheme
            BasicTextField(text, { text = it }, Modifier.fillMaxWidth(), maxLines = 6,
                textStyle = TextStyle(fontSize = 16.sp, color = scheme.onSurface), cursorBrush = SolidColor(scheme.primary),
                decorationBox = { inner ->
                    Box { if (text.isEmpty()) Text("Message", fontSize = 16.sp, color = scheme.onSurfaceVariant); inner() }
                })
            Spacer(Modifier.height(6.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                SmallIcon(AppIcons.Add, "Attach a picture", enabled = !busy) {
                    pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                }
                SmallIcon(AppIcons.Think, if (thinking) "Think first: on" else "Think first: off", enabled = !busy,
                    tint = if (thinking) scheme.primary else scheme.onSurfaceVariant) {
                    if (conv != null) chat.setThinking(conv.id, !thinking) else newThinking = !thinking
                }
                Spacer(Modifier.width(4.dp))
                ModelPill(slots, enabled = !busy) { n -> slots = n; LocalLing.setExpertSlots(ctx, n) }
                Spacer(Modifier.weight(1f))
                if (mine) FilledTonalIconButton(onClick = { chat.stop() }, Modifier.size(40.dp)) {
                    Icon(AppIcons.Stop, "Stop", Modifier.size(20.dp))
                } else FilledIconButton(onClick = {
                    val id = chat.send(convId, text.trim(), attached, thinking)
                    open(id); text = ""; attached = null
                }, Modifier.size(40.dp), enabled = !busy && (text.isNotBlank() || attached != null)) {
                    Icon(AppIcons.Send, "Send", Modifier.size(20.dp))
                }
            }
        }
    }
}

/** The bar on top: the side menu, a title with a line under it, and the page's own buttons. */
@Composable
fun ChatBar(onMenu: () -> Unit, title: String, subtitle: String = "", buttons: Int = 0,
            actions: (@Composable () -> Unit)? = null) {
    Row(Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = onMenu) { Icon(AppIcons.Menu, "Menu") }
        // as wide on the left as the page's buttons on the right, so the title sits in the middle
        if (buttons > 1) Spacer(Modifier.width((48 * (buttons - 1)).dp))
        Column(Modifier.weight(1f).padding(horizontal = 4.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (subtitle.isNotEmpty()) Text(subtitle, fontSize = 11.5.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        // a page without buttons of its own keeps the title centred
        if (actions != null) Row { actions() } else Spacer(Modifier.width(48.dp))
    }
}

// the chat model choices, as in Settings (setting value, pill text, menu line)
private val CHAT_MODELS = listOf(
    Triple(0, "Ling · Fastest", "about 4.9 GB, quickest"),
    Triple(48, "Ling · Balanced", "about 2.6 GB"),
    Triple(32, "Ling · Low", "about 2.3 GB"),
    Triple(LocalLing.SMALL, "Qwen3.5 0.8B", "about 1.4 GB, less clever"),
)

private fun modelLabel(ctx: Context, slots: Int) =
    CHAT_MODELS.firstOrNull { it.first == slots }?.second ?: LocalLing.name(ctx)

/** Which model answers: a pill in the composer, the way chat apps place it. */
@Composable
private fun ModelPill(slots: Int, enabled: Boolean, onPick: (Int) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        Surface(Modifier.clip(RoundedCornerShape(18.dp)).clickable(enabled = enabled) { open = true },
            shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceContainerHighest) {
            Text(CHAT_MODELS.firstOrNull { it.first == slots }?.second ?: "Ling", Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                fontSize = 13.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
        DropdownMenu(open, { open = false }) {
            CHAT_MODELS.forEach { (n, label, about) ->
                DropdownMenuItem(text = {
                    Column {
                        Text(label, fontWeight = if (n == slots) FontWeight.SemiBold else FontWeight.Normal)
                        Text(about, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }, onClick = { open = false; if (n != slots) onPick(n) })
            }
        }
    }
}

@Composable
private fun SmallIcon(icon: ImageVector, what: String, enabled: Boolean = true,
                      tint: androidx.compose.ui.graphics.Color = MaterialTheme.colorScheme.onSurfaceVariant, size: Int = 22,
                      onClick: () -> Unit) {
    IconButton(onClick = onClick, Modifier.size(40.dp), enabled = enabled) {
        Icon(icon, what, Modifier.size(size.dp), tint = if (enabled) tint else tint.copy(alpha = 0.4f))
    }
}

@Composable
private fun StatusLine(status: String) {
    Row(Modifier.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.width(10.dp))
        Text(status.ifEmpty { "Working…" }, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

private fun copy(ctx: Context, text: String) {
    (ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("chat", text))
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun MessageRow(m: Msg, dir: File, onOpen: (Gallery.Entry) -> Unit) {
    val ctx = LocalContext.current
    val scheme = MaterialTheme.colorScheme
    when (m) {
        is Msg.User -> Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.End) {
            m.imageFile?.let { f ->
                val b by produceState<Bitmap?>(null, f) {
                    value = withContext(Dispatchers.IO) { decodeSmall(File(dir, f), 320) }
                }
                b?.let { Image(it.asImageBitmap(), null, Modifier.padding(bottom = 6.dp).size(140.dp).clip(RoundedCornerShape(14.dp)),
                    contentScale = ContentScale.Crop) }
            }
            if (m.text.isNotBlank()) Text(m.text,
                Modifier.widthIn(max = (LocalConfiguration.current.screenWidthDp * 0.82f).dp)
                    .clip(RoundedCornerShape(18.dp)).background(scheme.surfaceContainerHighest)
                    .combinedClickable(onClick = {}, onLongClick = { copy(ctx, m.text) })
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                fontSize = 15.sp, lineHeight = 21.sp)
        }
        is Msg.Assistant -> Reply(m.text, m.thinking, actions = true)
        is Msg.ToolCall -> {
            val a = runCatching { org.json.JSONObject(m.argsJson) }.getOrNull()
            val edit = a?.optBoolean("edit") == true
            val model = when (a?.optString("model")) { "anima" -> if (edit) "Qwen" else "Anima"; "qwen" -> "Qwen"; else -> "Picture" }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(AppIcons.Brush, null, Modifier.size(14.dp), tint = scheme.onSurfaceVariant)
                Spacer(Modifier.width(6.dp))
                Text((if (edit) "$model edit · " else "$model · ") + (a?.optString("prompt") ?: m.argsJson),
                    fontSize = 12.5.sp, color = scheme.onSurfaceVariant, maxLines = 2, overflow = TextOverflow.Ellipsis)
            }
        }
        is Msg.ToolResult -> {
            val err = runCatching { org.json.JSONObject(m.resultJson).optString("error") }.getOrNull()
            if (!err.isNullOrEmpty()) Text("Not queued: $err", fontSize = 12.5.sp, color = scheme.error)
        }
        is Msg.Picture -> PictureCard(m.jobId, onOpen)
    }
}

/** A reply: plain text across the page (selectable), its thinking folded above, a copy button under it. */
@Composable
private fun Reply(text: String, thinking: String, actions: Boolean) {
    val ctx = LocalContext.current
    var open by remember { mutableStateOf(false) }
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Column(Modifier.fillMaxWidth()) {
        if (thinking.isNotBlank()) {
            Text(if (open) "Thought ▾" else "Thought ▸", fontSize = 12.5.sp, color = muted,
                modifier = Modifier.clickable { open = !open }.padding(vertical = 2.dp))
            if (open) Text(thinking, Modifier.padding(start = 10.dp, bottom = 6.dp), fontSize = 13.sp, lineHeight = 18.sp, color = muted)
        }
        if (text.isNotBlank()) {
            SelectionContainer { Text(text, fontSize = 15.sp, lineHeight = 21.sp) }
            if (actions) IconButton(onClick = { copy(ctx, text) }, Modifier.size(32.dp)) {
                Icon(AppIcons.Copy, "Copy", Modifier.size(17.dp), tint = muted)
            }
        }
    }
}

@Composable
private fun RenameDialog(title: String, onDone: (String?) -> Unit) {
    var t by remember { mutableStateOf(title) }
    AlertDialog(onDismissRequest = { onDone(null) }, title = { Text("Rename chat") },
        text = { OutlinedTextField(t, { t = it }, singleLine = true) },
        confirmButton = { TextButton(onClick = { onDone(t.trim()) }) { Text("Save") } },
        dismissButton = { TextButton(onClick = { onDone(null) }) { Text("Cancel") } })
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
    Box(Modifier.widthIn(max = 280.dp).fillMaxWidth().clip(RoundedCornerShape(14.dp))
        .background(MaterialTheme.colorScheme.surfaceContainerHigh)) {
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
            Row(Modifier.padding(horizontal = 14.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                if (j?.state == JobState.WAITING || j?.state == JobState.DRAWING) {
                    CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Spacer(Modifier.width(10.dp))
                }
                Text(what, Modifier.weight(1f), fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (j?.state == JobState.FAILED || j?.state == JobState.CANCELLED) IconButton(onClick = {
                    Queue.store.update(j.id) { it.copy(state = JobState.WAITING, error = null, tries = 0) }
                    QueueService.kick(ctx)
                }, Modifier.size(32.dp)) { Icon(AppIcons.Refresh, "Retry", Modifier.size(18.dp)) }
            }
        }
    }
}

@Composable
private fun ChatList(chat: ChatController, current: String?, onOpen: (String?) -> Unit, onBack: () -> Unit) {
    val tick by chat.changed.collectAsState()
    val all = remember(tick) { chat.store().list() }
    val fmt = remember { DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT) }
    val scheme = MaterialTheme.colorScheme
    Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) { Icon(AppIcons.Back, "Back") }
            Text("Chats", Modifier.weight(1f), fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
            IconButton(onClick = { onOpen(null) }, enabled = !chat.busy()) { Icon(AppIcons.Add, "New chat") }
        }
        if (all.isEmpty()) Text("No chats yet.", Modifier.padding(20.dp), color = scheme.onSurfaceVariant)
        LazyColumn {
            items(all, key = { it.id }) { c ->
                Row(Modifier.fillMaxWidth().clickable { onOpen(c.id) }
                    .background(if (c.id == current) scheme.surfaceContainerHigh else scheme.surface)
                    .padding(start = 20.dp, end = 8.dp, top = 10.dp, bottom = 10.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(c.title, maxLines = 1, overflow = TextOverflow.Ellipsis, fontSize = 15.sp)
                        Text(fmt.format(Date(c.updated)), fontSize = 12.sp, color = scheme.onSurfaceVariant)
                    }
                    IconButton(onClick = {
                        chat.store().delete(c.id); chat.changed.value++
                        if (c.id == current) onOpen(null)
                    }, enabled = !(chat.busy() && chat.state.value.convId == c.id)) {
                        Icon(AppIcons.Delete, "Delete", Modifier.size(20.dp), tint = scheme.onSurfaceVariant)
                    }
                }
                HorizontalDivider(Modifier.padding(start = 20.dp), color = scheme.outlineVariant.copy(alpha = 0.4f))
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
