package com.anima.npu

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.runtime.collectAsState
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Icon
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.rememberDrawerState
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.Switch
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.random.Random

/** What the Create screen keeps across tab switches. */
class CreateState {
    // Pictures run here rather than in the Create screen's own scope, which is cancelled when the screen is left
    // (Gallery tab, a picture's details): the drawing finished but its result was dropped and the screen stayed busy.
    val work = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    var model by mutableStateOf("anima")  // ModelStore id of the model that draws
    var qwenDrawing by mutableStateOf("qwen")  // Qwen's drawing model: "qwen" (Qwen-Image 2.1) or "noctq" (qwen/noctq/)
    var animaPrompt by mutableStateOf("masterpiece, best quality, safe, 1girl, solo, long hair, smile, cherry blossoms, outdoors")
    var kleinPrompt by mutableStateOf("a cozy coffee shop interior with plants, warm light, people chatting, film photography")
    var qwenPrompt by mutableStateOf("A digital painting of a knight in silver armor holding a glowing sword in a dark forest, dramatic lighting.")
    var prompt: String
        get() = when (model) { "klein" -> kleinPrompt; "qwen" -> qwenPrompt; else -> animaPrompt }
        set(v) { when (model) { "klein" -> kleinPrompt = v; "qwen" -> qwenPrompt = v; else -> animaPrompt = v } }
    var edit by mutableStateOf<Bitmap?>(null)  // Klein / Qwen: the picture to change (cropped per size); null = new picture
    var seed by mutableStateOf(Random.nextLong(0, 1_000_000_000))
    var seedLocked by mutableStateOf(false)  // the user typed a seed: keep it for the next jobs too
    var shownJob by mutableStateOf<String?>(null)  // the finished queue job shown on the Create screen
    var steps by mutableIntStateOf(8)
    var busy by mutableStateOf(false)
    var status by mutableStateOf("")
    var picture by mutableStateOf<Bitmap?>(null)
    var pictureMs by mutableStateOf(0L)
    var savedFor by mutableStateOf<Bitmap?>(null)  // the picture already saved, so Save is not offered twice
    var memoryMode by mutableIntStateOf(1)
    var tier by mutableIntStateOf(0)    // index into SIZE_TIERS
    var shape by mutableIntStateOf(0)   // index into SHAPES
    var guide by mutableStateOf<Bitmap?>(null)  // the guide picture as picked (shown small, cropped per size)
    var guideStrength by mutableStateOf(1f)
    var guideEnd by mutableStateOf(0.5f)
    var genW by mutableIntStateOf(512)  // size of the last generated picture (the upscale input)
}

// Picture sizes the drawing model is built for: per tier, one size per shape. Width/16 x height/16 (the patch
// count) must divide by 16 - other counts compile into ~4x slower programs on the neural unit.
// Times are 8 steps on the iQOO 13 (8 Elite).
val SIZE_TIERS = listOf("512" to "about 6 s", "768" to "about 17 s")
const val KLEIN_TIME = "4 steps · about 6 s, edits about 13 s"
const val QWEN_TIME = "4 steps · about 1 min 20 s, edits about 3 min"
/** Folder (inside Qwen's) holding Noct Q V4's drawing files, copied over the cable: no download for it. */
const val NOCTQ_DIR = "noctq"
val SHAPES = listOf("Square", "Tall 9:16", "Wide 16:9")
private val SIZES = listOf(
    listOf(512 to 512, 384 to 672, 672 to 384),
    listOf(768 to 768, 576 to 1024, 1024 to 576),
)
fun pictureSize(tier: Int, shape: Int): Pair<Int, Int> = SIZES[tier][shape]

/** [tier, shape] of a picture size, or null if it is not one of ours. */
fun sizeIndex(w: Int, h: Int): Pair<Int, Int>? {
    SIZES.forEachIndexed { t, row -> row.forEachIndexed { k, wh -> if (wh == w to h) return t to k } }
    return null
}

/** The guide picture cropped to the picture's shape (centre) and scaled to its size, as ARGB pixels. */
fun guidePixels(src: Bitmap, w: Int, h: Int): IntArray {
    val scale = maxOf(w.toFloat() / src.width, h.toFloat() / src.height)
    val cw = (w / scale).toInt().coerceAtMost(src.width)
    val ch = (h / scale).toInt().coerceAtMost(src.height)
    val crop = Bitmap.createBitmap(src, (src.width - cw) / 2, (src.height - ch) / 2, cw, ch)
    val scaled = Bitmap.createScaledBitmap(crop, w, h, true)
    return IntArray(w * h).also { scaled.getPixels(it, 0, w, 0, 0, w, h) }
}

class MainActivity : ComponentActivity() {
    // a finished picture to open, from a "Picture ready" notification
    private val openUri = mutableStateOf<String?>(null)

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        intent.getStringExtra("openUri")?.let { openUri.value = it }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = getExternalFilesDir(null)!!
        val modelDir = File(root, "models")
        // The app must create these folders itself; one made over the cable first is not readable by the app.
        modelDir.mkdirs()
        ModelStore.ALL.forEach { it.modelDir(this).mkdirs() }
        val libDir = applicationInfo.nativeLibraryDir
        val selfTest = intent.getBooleanExtra("selftest", false)
        val loadOpts = intent.getIntExtra("loadopts", -1)
        val memOverride = intent.getIntExtra("memmode", -1)
        Engines.init(this, if (memOverride >= 0) memOverride else if (selfTest) 1 else -1, loadOpts)
        Queue.init(this)
        openUri.value = intent.getStringExtra("openUri")
        if (checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 1)
        // jobs left waiting (or interrupted) when the app was last closed
        if (!selfTest && Queue.store.waitingCount() > 0) QueueService.kick(this)
        // test builds: `am start ... --es chat "<message>" [--ez think true] [--es conv <id>] [--es image <file in
        // the app's files folder>]` sends a chat message. Only when files/test-shortcut exists (made over the cable
        // with run-as): published builds are debug builds too, and other apps can start this screen.
        if (BuildConfig.DEBUG && File(filesDir, "test-shortcut").exists()) intent.getStringExtra("chat")?.let { msg ->
            val img = intent.getStringExtra("image")?.let { android.graphics.BitmapFactory.decodeFile(File(filesDir, it).absolutePath) }
            Chat.get(this).send(intent.getStringExtra("conv"), msg, img, intent.getBooleanExtra("think", false))
        }

        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                val ctx = this@MainActivity
                // installed variant per model (ModelStore id -> variant id or null)
                var installedOf by remember { mutableStateOf(ModelStore.ALL.associate { it.id to it.installed(ctx) }) }
                val installed = installedOf["anima"]
                var downloadFor by remember { mutableStateOf<String?>(null) }  // ModelStore id whose download screen is open
                var updating by remember { mutableStateOf(false) }
                var updateBytesOf by remember { mutableStateOf(mapOf<String, Long>()) }
                // Quietly look for newer model files once per launch (the lists are tiny).
                LaunchedEffect(installedOf) {
                    if (!selfTest) for (st in ModelStore.ALL) if (installedOf[st.id] != null) runCatching {
                        val m = st.fetchManifest(ctx)
                        updateBytesOf = updateBytesOf + (st.id to st.outdated(ctx, m).sumOf { it.size })
                    }
                }
                // a finished download, wherever the user is: the engine reads the new files (sizes, tables) afresh
                LaunchedEffect(Installer.finished) {
                    val (sid, v) = Installer.finished ?: return@LaunchedEffect
                    Installer.finished = null
                    installedOf = installedOf + (sid to v); downloadFor = null; updating = false
                    updateBytesOf = updateBytesOf + (sid to 0L)
                    Engines.invalidate(sid)
                }
                val prefs = remember { getSharedPreferences("anima", MODE_PRIVATE) }
                val create = remember {
                    CreateState().also {
                        it.memoryMode = prefs.getInt("memoryMode", 1)
                        it.tier = prefs.getInt("tier", 0).coerceIn(0, SIZE_TIERS.size - 1)
                        it.shape = prefs.getInt("shape", 0).coerceIn(0, SHAPES.size - 1)
                        it.model = prefs.getString("model", "anima") ?: "anima"
                        it.qwenDrawing = prefs.getString("qwenDrawing", "qwen") ?: "qwen"
                    }
                }
                var tab by remember { mutableIntStateOf(0) }
                var open by remember { mutableStateOf<Gallery.Entry?>(null) }
                var showQueue by remember { mutableStateOf(false) }
                LaunchedEffect(openUri.value) {
                    val uri = openUri.value ?: return@LaunchedEffect
                    openUri.value = null
                    withContext(Dispatchers.IO) { Gallery.list(ctx).firstOrNull { it.uri.toString() == uri } }?.let { open = it }
                }

                if (selfTest) {
                    LaunchedEffect(Unit) {
                        Engines.ensureAnima()?.let { create.status = it; return@LaunchedEffect }
                        withContext(Dispatchers.Default) { SelfTest.run(root) { create.status = it } }
                    }
                }

                val drawer = rememberDrawerState(DrawerValue.Closed)
                val navScope = rememberCoroutineScope()
                val openMenu: () -> Unit = { navScope.launch { drawer.open() } }
                // The pages live in a side menu, as in sysmon's phone app, instead of a bar along the bottom that
                // took a strip of every screen (2026-10-09).
                fun go(t: Int) { downloadFor = null; updating = false; showQueue = false; open = null; tab = t; navScope.launch { drawer.close() } }
                ModalNavigationDrawer(drawerState = drawer, gesturesEnabled = open == null, drawerContent = {
                    ModalDrawerSheet(Modifier.safeDrawingPadding()) {
                        Text("Anima", Modifier.padding(start = 28.dp, top = 20.dp, bottom = 16.dp),
                            style = MaterialTheme.typography.titleLarge)
                        val qTick by Queue.store.changed.collectAsState()
                        val waiting = remember(qTick) { Queue.store.waitingCount() }
                        @Composable fun item(label: String, icon: androidx.compose.ui.graphics.vector.ImageVector, selected: Boolean,
                                             badge: Int = 0, onClick: () -> Unit) =
                            NavigationDrawerItem(label = { Text(label) }, selected = selected, onClick = onClick,
                                icon = { Icon(icon, null) }, badge = { if (badge > 0) Text("$badge") },
                                modifier = Modifier.padding(horizontal = 12.dp))
                        val here = downloadFor == null && open == null && !showQueue
                        item("Create", AppIcons.Create, here && tab == 0) { go(0) }
                        item("Chat", AppIcons.Chat, here && tab == 1) { go(1) }
                        item("Gallery", AppIcons.Gallery, here && tab == 2) { go(2) }
                        item("Queue", AppIcons.Queue, showQueue, badge = waiting) { go(0); showQueue = true }
                        item("Settings", AppIcons.Settings, here && tab == 3) { go(3) }
                    }
                }) {
                Scaffold(
                    modifier = Modifier.safeDrawingPadding(),
                    topBar = {
                        Column {
                            AppUpdate.available?.let { u ->
                                Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                                    Text("Update available" + (u.notes.takeIf { it.isNotBlank() }?.let { ": $it" } ?: ""),
                                        style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                                    TextButton(onClick = { lifecycleScope.launch { AppUpdate.install(ctx) {
                                        when {
                                            QueueService.running -> "A picture is drawing"
                                            Chat.busy() -> "${LocalLing.name(ctx)} is answering"
                                            else -> null
                                        }
                                    } } },
                                        enabled = !AppUpdate.busy) { Text("Update") }
                                }
                                if (AppUpdate.status.isNotEmpty()) Text(AppUpdate.status, style = MaterialTheme.typography.bodySmall,
                                    modifier = Modifier.padding(horizontal = 16.dp))
                            }
                            UsageMonitor()
                        }
                    },
                ) { pad ->
                    Box(Modifier.padding(pad).fillMaxSize()) {
                        when {
                            ModelStore.PICTURE.all { installedOf[it.id] == null } && !selfTest || downloadFor != null -> {
                                val st = ModelStore.byId(downloadFor ?: "anima")
                                DownloadScreen(
                                    store = st,
                                    current = installedOf[st.id],
                                    beforeSwap = { Engine.release() },
                                    onBack = if (ModelStore.PICTURE.any { installedOf[it.id] != null }) ({ downloadFor = null; updating = false }) else null,
                                    update = updating,
                                )
                            }
                            showQueue && open == null -> QueueScreen(onOpen = { open = it }, onBack = { showQueue = false })
                            open != null -> Detail(open!!, onBack = { open = null }, onUse = { e ->
                                create.prompt = e.prompt; create.seed = e.seed; if (create.model == "anima") create.steps = e.steps
                                sizeIndex(e.width, e.height)?.let { (t, k) -> create.tier = t; create.shape = k }
                                open = null; tab = 0
                            })
                            tab == 0 -> Column {
                                ChatBar(openMenu, "Create")
                                val st = ModelStore.byId(create.model)
                                val inst = installedOf[st.id]
                                Create(create, Engines::ensureAnima, Engines::ensureKlein, { Engines.ensureQwen(create.qwenDrawing) },
                                    modelLabel = st.title + (st.cachedManifest(ctx)?.variants?.firstOrNull { it.id == inst }?.label?.let { " · $it" } ?: ""),
                                    installed = inst != null, animaInstalled = installed != null,
                                    onChangeModel = { downloadFor = st.id }, onDownload = { downloadFor = st.id },
                                    updateBytes = updateBytesOf[st.id] ?: 0L, onUpdate = { updating = true; downloadFor = st.id },
                                    onModel = { id ->
                                        if (id != create.model) Engine.release()  // the other model's pieces leave the neural unit
                                        create.model = id
                                        prefs.edit().putString("model", id).apply()
                                    },
                                    onSize = { prefs.edit().putInt("tier", create.tier).putInt("shape", create.shape).apply() },
                                    onQueue = { showQueue = true },
                                    onQwenDrawing = { id ->
                                        if (id != create.qwenDrawing) {
                                            create.qwenDrawing = id
                                            prefs.edit().putString("qwenDrawing", id).apply()
                                        }
                                    }) { m ->
                                    create.memoryMode = m
                                    prefs.edit().putInt("memoryMode", m).apply()
                                    Engines.setMemoryMode(m)
                                }
                            }
                            tab == 1 -> ChatScreen(Chat.get(ctx), isInstalled = { installedOf[it] != null },
                                onDownload = { downloadFor = LocalLing.store(ctx).id }, onOpen = { open = it }, onMenu = openMenu)
                            tab == 3 -> Column { ChatBar(openMenu, "Settings"); SettingsScreen() }
                            else -> Column { ChatBar(openMenu, "Gallery"); GalleryGrid(onOpen = { open = it }) }
                        }
                    }
                }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Look for a newer build of the app each time it comes to the front (a tiny file, through DownloadManager).
        lifecycleScope.launch { AppUpdate.check(this@MainActivity) }
        // jobs added while the app was in the background (Android refuses to start the worker then)
        if (Queue.store.waitingCount() > 0 && !QueueService.running) QueueService.kick(this)
    }

    override fun onStop() {
        super.onStop()
        // Give the memory back when the app is not on screen (the next picture reloads, a second or two), unless
        // the queue is drawing in the background.
        if (!QueueService.running) Engines.release()
        // Ling holds about 4.5 GB; it starts again (about 10 s) with the next message
        if (!QueueService.running && !Chat.busy() && LocalLing.running()) Thread { LocalLing.stop() }.start()
    }
}

// Measured on the iQOO 13 (8 Elite), 8 steps, 2026-10-07.
private val MEMORY_MODES = listOf(
    "Fastest" to "keeps everything loaded",
    "Normal" to "loads in turns",
    "Low" to "swaps half the drawing model in and out",
    "Lowest" to "one piece of the drawing model at a time",
)

@Composable
fun Create(
    s: CreateState, ensureEngine: suspend () -> String?, ensureKlein: suspend () -> String?, ensureQwen: suspend () -> String?,
    modelLabel: String,
    installed: Boolean, animaInstalled: Boolean, onChangeModel: () -> Unit, onDownload: () -> Unit,
    updateBytes: Long, onUpdate: () -> Unit, onModel: (String) -> Unit, onSize: () -> Unit,
    onQwenDrawing: (String) -> Unit, onQueue: () -> Unit, onMemoryMode: (Int) -> Unit,
) {
    val ctx = LocalContext.current
    val scope = s.work  // outlives this screen: a tab switch must not drop a picture being drawn
    val klein = s.model == "klein"
    val qwen = s.model == "qwen"
    val anima = !klein && !qwen
    // the Small Qwen download reads prompts with the 2B and cannot edit pictures
    val qwenSmall = qwen && remember(installed) { ModelStore.QWEN.installed(ctx) == "small" }
    val read = { uri: Uri -> runCatching { ctx.contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it) } }.getOrNull() }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        if (uri != null) scope.launch(Dispatchers.IO) {
            s.guide = read(uri)
            if (s.guide == null) s.status = "Could not read that picture"
        }
    }
    val pickEdit = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        if (uri != null) scope.launch(Dispatchers.IO) {
            s.edit = read(uri)
            if (s.edit == null) s.status = "Could not read that picture"
        }
    }
    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        SingleChoiceSegmentedButtonRow {
            ModelStore.PICTURE.forEachIndexed { i, st ->
                SegmentedButton(selected = s.model == st.id, onClick = { onModel(st.id) }, enabled = !s.busy,
                    shape = SegmentedButtonDefaults.itemShape(i, ModelStore.PICTURE.size)) {
                    Text(when (st.id) { "klein" -> "Klein"; "qwen" -> "Qwen"; else -> st.title })
                }
            }
        }
        if (!installed) {
            Text("${ModelStore.byId(s.model).title} is not downloaded yet." + when {
                klein -> " It draws from plain sentences and can edit your pictures (about 4 GB)."
                qwen -> " It draws 768×768 pictures from plain sentences and can edit your pictures (about 15 GB)."
                else -> ""
            })
            Button(onClick = onDownload, enabled = !Installer.running) { Text("Download…") }
            if (Installer.running) Text("Another download is running: ${Installer.status}", style = MaterialTheme.typography.bodySmall)
            return@Column
        }
        OutlinedTextField(s.prompt, { s.prompt = it }, Modifier.fillMaxWidth(),
            label = { Text(if (!anima && !qwenSmall && s.edit != null) "What to change" else "Prompt") }, minLines = 3, enabled = !s.busy)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                s.seed.toString(), { v -> v.toLongOrNull()?.let { s.seed = it; s.seedLocked = true } }, Modifier.width(170.dp),
                label = { Text("Seed") }, singleLine = true, enabled = !s.busy,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            )
            OutlinedButton(onClick = { s.seed = Random.nextLong(0, 1_000_000_000); s.seedLocked = false }, enabled = !s.busy) { Text("🎲") }
        }
        if (anima) {
            Text("Steps (fewer is faster, 8 is the full quality)")
            SingleChoiceSegmentedButtonRow {
                listOf(4, 6, 8).forEachIndexed { i, n ->
                    SegmentedButton(selected = s.steps == n, onClick = { s.steps = n }, enabled = !s.busy,
                        shape = SegmentedButtonDefaults.itemShape(i, 3)) { Text("$n") }
                }
            }
        }
        if (!qwen) Text("Size")
        if (anima) SingleChoiceSegmentedButtonRow {
            SIZE_TIERS.forEachIndexed { i, (name, _) ->
                SegmentedButton(selected = s.tier == i, onClick = { s.tier = i; onSize() }, enabled = !s.busy,
                    shape = SegmentedButtonDefaults.itemShape(i, SIZE_TIERS.size)) { Text(name) }
            }
        }
        if (!qwen) SingleChoiceSegmentedButtonRow {
            SHAPES.forEachIndexed { i, name ->
                SegmentedButton(selected = s.shape == i, onClick = { s.shape = i; onSize() }, enabled = !s.busy,
                    shape = SegmentedButtonDefaults.itemShape(i, SHAPES.size)) { Text(name, maxLines = 1) }
            }
        }
        val (pw, ph) = if (qwen) 768 to 768 else pictureSize(if (klein) 0 else s.tier, s.shape)
        Text("$pw×$ph · " + when { klein -> KLEIN_TIME; qwen -> QWEN_TIME; else -> SIZE_TIERS[s.tier].second },
            style = MaterialTheme.typography.bodySmall)
        if (qwenSmall) Text("Editing pictures needs the Standard Qwen download (Model: Change).", style = MaterialTheme.typography.bodySmall)
        // Noct Q: a fine-tune of Qwen-Image 2.1 (uncensored), copied to the phone into qwen/noctq/ - only its drawing
        // files, the readers and decoder are Qwen's
        val noctq = qwen && remember(installed) { File(ModelStore.QWEN.modelDir(ctx), "$NOCTQ_DIR/sp31.bin").isFile }
        if (noctq) {
            Text("Drawing model", style = MaterialTheme.typography.bodySmall)
            SingleChoiceSegmentedButtonRow {
                listOf("qwen" to "Qwen-Image", "noctq" to "Noct Q").forEachIndexed { i, (id, name) ->
                    SegmentedButton(selected = s.qwenDrawing == id, onClick = { onQwenDrawing(id) }, enabled = !s.busy,
                        shape = SegmentedButtonDefaults.itemShape(i, 2)) { Text(name, maxLines = 1) }
                }
            }
        }
        if (!anima && !qwenSmall) {
            Text("Picture to edit (optional): ${if (klein) "Klein" else "Qwen"} changes it as you describe")
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                s.edit?.let { Image(it.asImageBitmap(), null, Modifier.size(72.dp), contentScale = ContentScale.Crop) }
                OutlinedButton(onClick = { pickEdit.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                    enabled = !s.busy) { Text(if (s.edit == null) "Choose…" else "Change") }
                if (s.edit != null) TextButton(onClick = { s.edit = null }, enabled = !s.busy) { Text("Remove") }
            }
            if (s.edit != null) Text((if (qwen) "It is cropped to a square." else "It is cropped to the shape above.") +
                " Describe the change, e.g. \"make her hair red\".", style = MaterialTheme.typography.bodySmall)
        }
        if (anima) Text("Guide picture (optional): the layout and pose follow it, the prompt decides the rest")
        if (anima) Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            s.guide?.let { Image(it.asImageBitmap(), null, Modifier.size(72.dp), contentScale = ContentScale.Crop) }
            OutlinedButton(onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                enabled = !s.busy) { Text(if (s.guide == null) "Choose…" else "Change") }
            if (s.guide != null) TextButton(onClick = { s.guide = null }, enabled = !s.busy) { Text("Remove") }
        }
        if (anima && s.guide != null) {
            Text("How closely to follow it: %.1f".format(s.guideStrength), style = MaterialTheme.typography.bodySmall)
            Slider(s.guideStrength, { s.guideStrength = (it * 10).toInt() / 10f }, valueRange = 0.2f..1.5f, enabled = !s.busy)
            Text("For how much of the drawing: %d%%".format((s.guideEnd * 100).toInt()), style = MaterialTheme.typography.bodySmall)
            Slider(s.guideEnd, { s.guideEnd = (it * 20).toInt() / 20f }, valueRange = 0.2f..1f, enabled = !s.busy)
            Text("Less of the drawing keeps the pose but lets the prompt change clothes and details.",
                style = MaterialTheme.typography.bodySmall)
        }
        if (!qwen) Text("Memory use")
        if (!qwen) SingleChoiceSegmentedButtonRow {
            MEMORY_MODES.forEachIndexed { i, (name, _) ->
                SegmentedButton(selected = s.memoryMode == i, onClick = { onMemoryMode(i) }, enabled = !s.busy,
                    shape = SegmentedButtonDefaults.itemShape(i, MEMORY_MODES.size)) { Text(name) }
            }
        }
        if (!qwen) Text(MEMORY_MODES[s.memoryMode].second, style = MaterialTheme.typography.bodySmall)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Model: $modelLabel", style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = onChangeModel, enabled = !s.busy) { Text("Change") }
        }
        if (Installer.running) {
            Text("Updating the model: ${Installer.status}", style = MaterialTheme.typography.bodySmall)
            LinearProgressIndicator(progress = { Installer.fraction }, modifier = Modifier.fillMaxWidth())
        } else if (updateBytes > 0) Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Model update available (%.1f GB)".format(updateBytes / 1e9), style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = onUpdate, enabled = !s.busy) { Text("Update") }
        }
        Button(onClick = {
            enqueueFromCreate(ctx, s)
            if (!s.seedLocked) s.seed = Random.nextLong(0, 1_000_000_000)   // the next job draws a new picture
            s.status = "Added to the queue"
        }, enabled = !Installer.running && s.prompt.isNotBlank() && installed, modifier = Modifier.fillMaxWidth()) {
            Text(if (Installer.running) "Add to queue (after the update)" else "Add to queue")
        }
        // the newest finished picture from this screen, and how the queue is doing
        val tick by Queue.store.changed.collectAsState()
        val progress by QueueService.progress.collectAsState()
        val queueLine = remember(tick, progress) {
            val left = Queue.store.waitingCount()
            if (left == 0) "" else "In the queue: $left" + (Queue.store.get(progress.first)?.let { j ->
                " · drawing \"${j.prompt.take(30)}…\"" + if (progress.second.isNotEmpty()) ": ${progress.second}" else "" } ?: "")
        }
        LaunchedEffect(tick) {
            val j = Queue.store.all().lastOrNull { it.origin == "create" && it.state == JobState.DONE && it.resultUri != null }
                ?: return@LaunchedEffect
            if (j.id == s.shownJob) return@LaunchedEffect
            val e = Gallery.Entry(android.net.Uri.parse(j.resultUri), j.prompt, j.seed, j.steps, j.ms, j.width, 0, j.height)
            val bmp = withContext(Dispatchers.IO) { Gallery.full(ctx, e) } ?: return@LaunchedEffect
            s.shownJob = j.id
            s.picture = bmp; s.savedFor = bmp; s.genW = bmp.width; s.pictureMs = j.ms
            s.status = "Done in %.0f s · saved to the gallery".format(j.ms / 1000.0)
        }
        if (queueLine.isNotEmpty()) TextButton(onClick = onQueue) { Text(queueLine, style = MaterialTheme.typography.bodySmall) }
        if (s.status.isNotEmpty()) Text(s.status, style = MaterialTheme.typography.bodySmall)
        s.picture?.let { bmp ->
            Image(bmp.asImageBitmap(), null, Modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    scope.launch(Dispatchers.IO) {
                        Gallery.save(ctx, bmp, s.prompt, s.seed, if (anima) s.steps else 4, s.pictureMs)
                        s.savedFor = bmp
                        s.status = "Saved to Pictures/Anima (${bmp.width}×${bmp.height})"
                    }
                }, enabled = !s.busy && s.savedFor !== bmp) { Text(if (s.savedFor === bmp) "Saved" else "Save") }
                if (bmp.width == s.genW && animaInstalled) OutlinedButton(onClick = {
                    s.busy = true
                    scope.launch {
                        val w = bmp.width
                        val h = bmp.height
                        val px = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }
                        val up = withContext(Dispatchers.Default) { ensureEngine(); Engine.upscale(px, w, h) }
                        if (up == null) s.status = Engine.lastError() else {
                            s.picture = Bitmap.createBitmap(up, 4 * w, 4 * h, Bitmap.Config.ARGB_8888)
                            s.status = "Upscaled to %d×%d in %.2f s".format(4 * w, 4 * h, Engine.lastTimes()[3] / 1000)
                        }
                        s.busy = false
                    }
                }, enabled = !s.busy && !Installer.running) { Text("Upscale 4×") }
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
fun GalleryGrid(onOpen: (Gallery.Entry) -> Unit) {
    val ctx = LocalContext.current
    var items by remember { mutableStateOf<List<Pair<Gallery.Entry, Bitmap?>>>(emptyList()) }
    var loaded by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        items = withContext(Dispatchers.IO) { Gallery.list(ctx).map { it to Gallery.thumbnail(ctx, it, 384) } }
        loaded = true
    }
    if (loaded && items.isEmpty()) {
        Text("No saved pictures yet.", Modifier.padding(16.dp))
        return
    }
    LazyVerticalGrid(GridCells.Fixed(3), Modifier.fillMaxSize().padding(4.dp)) {
        items(items, key = { it.first.uri.toString() }) { (e, thumb) ->
            Box(Modifier.padding(2.dp).aspectRatio(1f).clickable { onOpen(e) }) {
                thumb?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
            }
        }
    }
}

@Composable
fun Detail(e: Gallery.Entry, onBack: () -> Unit, onUse: (Gallery.Entry) -> Unit) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var bmp by remember { mutableStateOf<Bitmap?>(null) }
    BackHandler(onBack = onBack)
    LaunchedEffect(e) { bmp = withContext(Dispatchers.IO) { Gallery.full(ctx, e) } }
    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        TextButton(onClick = onBack) { Text("‹ Back") }
        bmp?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxWidth().aspectRatio(it.width.toFloat() / it.height)) }
        Text(e.prompt)
        Text("seed ${e.seed} · ${e.steps} steps · ${e.width}×${e.height}", style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { onUse(e) }) { Text("Use this prompt") }
            OutlinedButton(onClick = {
                val send = Intent(Intent.ACTION_SEND).setType("image/png").putExtra(Intent.EXTRA_STREAM, e.uri)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                ctx.startActivity(Intent.createChooser(send, "Share picture"))
            }) { Text("Share") }
            OutlinedButton(onClick = { scope.launch(Dispatchers.IO) { Gallery.delete(ctx, e); withContext(Dispatchers.Main) { onBack() } } }) { Text("Delete") }
        }
        // any picture as a home-screen icon: Android only lets the launcher icon itself be one of the built-in ones
        OutlinedButton(onClick = {
            val b = bmp ?: return@OutlinedButton
            if (!AppIcon.pinPicture(ctx, b, e.seed.toString() + "-" + e.uri.lastPathSegment))
                android.widget.Toast.makeText(ctx, "This launcher cannot add shortcuts", android.widget.Toast.LENGTH_SHORT).show()
        }, enabled = bmp != null) { Text("Add to home screen as an icon") }
    }
}
