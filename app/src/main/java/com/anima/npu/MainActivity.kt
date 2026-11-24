package com.anima.npu

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.border
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.material3.HorizontalDivider
import androidx.compose.foundation.layout.heightIn
import androidx.compose.ui.draw.rotate
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.NavigationRailItemDefaults
import androidx.compose.material3.VerticalDivider
import androidx.compose.material3.AlertDialog
import androidx.compose.ui.draw.clip
import androidx.compose.material3.IconButton
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.foundation.background
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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
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
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Badge
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.Switch
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.snapshotFlow
import kotlinx.coroutines.flow.debounce
import androidx.compose.runtime.produceState
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
    var advanced by mutableStateOf(false)   // "Steps, seed and memory" opened on Create
    // Pictures run here rather than in the Create screen's own scope, which is cancelled when the screen is left
    // (Gallery tab, a picture's details): the drawing finished but its result was dropped and the screen stayed busy.
    val work = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    var model by mutableStateOf("anima")  // ModelStore id of the model that draws
    var qwenDrawing by mutableStateOf("qwen")  // Qwen's drawing model: "qwen" (Qwen-Image 2.1) or "noctq" (qwen/noctq/)
    var animaPrompt by mutableStateOf("masterpiece, best quality, safe, 1girl, solo, long hair, smile, cherry blossoms, outdoors")
    var kleinPrompt by mutableStateOf("a cozy coffee shop interior with plants, warm light, people chatting, film photography")
    var kreaPrompt by mutableStateOf("a red fox sitting in fresh snow, morning light, photo")
    var qwenPrompt by mutableStateOf("A digital painting of a knight in silver armor holding a glowing sword in a dark forest, dramatic lighting.")
    var fiboPrompt by mutableStateOf("Make the sky stormy and dark")
    var mossPrompt by mutableStateOf("A lighthouse on a rocky cliff at sunset, waves crashing below, warm golden light.")
    var wanPrompt by mutableStateOf("She smiles and turns her head slowly toward the camera, her hair moving in a light breeze.")
    var prompt: String
        get() = when (model) { "klein" -> kleinPrompt; "krea" -> kreaPrompt; "qwen" -> qwenPrompt; "fibo" -> fiboPrompt; "moss" -> mossPrompt; "wan" -> wanPrompt; else -> animaPrompt }
        set(v) { when (model) { "klein" -> kleinPrompt = v; "krea" -> kreaPrompt = v; "qwen" -> qwenPrompt = v; "fibo" -> fiboPrompt = v; "moss" -> mossPrompt = v; "wan" -> wanPrompt = v; else -> animaPrompt = v } }
    /** Puts back each model's saved prompt; [saved] gives the stored text for a model id, or null for none. */
    fun restorePrompts(saved: (String) -> String?) {
        saved("anima")?.let { animaPrompt = it }; saved("klein")?.let { kleinPrompt = it }; saved("krea")?.let { kreaPrompt = it }; saved("qwen")?.let { qwenPrompt = it }
        saved("fibo")?.let { fiboPrompt = it }; saved("moss")?.let { mossPrompt = it }; saved("wan")?.let { wanPrompt = it }
    }
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
    var clipUri by mutableStateOf<Uri?>(null)  // the shown result is a Wan clip (an MP4): played instead of [picture]
}

/** Plays a clip from the gallery, looping, without sound: MP4 in a video view, an older GIF as an animated picture. */
@Composable
fun GifPlayer(uri: Uri, modifier: Modifier = Modifier) {
    val ctx = LocalContext.current
    val video = remember(uri) { runCatching { ctx.contentResolver.getType(uri) }.getOrNull()?.startsWith("video/") == true }
    if (video) {
        androidx.compose.ui.viewinterop.AndroidView(
            factory = { c -> android.widget.VideoView(c).apply {
                setOnPreparedListener { mp -> mp.isLooping = true; mp.setVolume(0f, 0f); start() }
                setVideoURI(uri)
            } }, modifier = modifier)
        return
    }
    var d by remember(uri) { mutableStateOf<android.graphics.drawable.Drawable?>(null) }
    LaunchedEffect(uri) {
        d = withContext(Dispatchers.IO) {
            runCatching { android.graphics.ImageDecoder.decodeDrawable(android.graphics.ImageDecoder.createSource(ctx.contentResolver, uri)) }.getOrNull()
        }
        (d as? android.graphics.drawable.AnimatedImageDrawable)?.apply { repeatCount = android.graphics.drawable.AnimatedImageDrawable.REPEAT_INFINITE; start() }
    }
    d?.let { dr ->
        androidx.compose.ui.viewinterop.AndroidView(
            factory = { c -> android.widget.ImageView(c).apply { adjustViewBounds = true; scaleType = android.widget.ImageView.ScaleType.FIT_CENTER } },
            update = { it.setImageDrawable(dr) }, modifier = modifier)
    }
}

// Picture sizes the drawing model is built for: per tier, one size per shape. Width/16 x height/16 (the patch
// count) must divide by 16 - other counts compile into ~4x slower programs on the neural unit.
// What each model makes, shown under the size choices (2026-11-24: no time estimates; speed differs too much
// between phones).
val SIZE_TIERS = listOf("512" to "", "768" to "")
const val KLEIN_TIME = "4 steps"
const val KLEIN_GPU_TIME = "8 steps on the graphics chip"
const val KLEIN_GEN5_TIME = "8 steps"   // 8 Elite Gen 5 AI-chip build, 768x768
const val QWEN_TIME = "8 steps"
const val NOCTQ_TIME = "4 steps"
/** Folder (inside Qwen's) holding Noct Q V4's drawing files, copied over the cable: no download for it. */
const val NOCTQ_DIR = "noctq"
const val FIBO_TIME = "edits only · 4 steps"
const val MOSS_TIME = "8 steps"
/** The models that draw one 768x768 size only, with no size or memory choices. */
val BIG_MODELS = setOf("qwen", "fibo", "moss", "wan", "krea")
const val WAN_TIME = "17 frames, a looping video"
const val WAN_LONG_TIME = "49 frames, a 3-second video"
const val WAN_GPU_TIME = "49 frames, a 3-second video, on the graphics chip"
const val MOSS_GPU_TIME = "8 steps on the graphics chip"
const val FIBO_GPU_TIME = "edits only · 4 steps on the graphics chip"

/**
 * The clip length the phone's Wan files make: on the graphics chip always 49 frames; on the AI chip frames.txt says 49
 * for the 3-second set, none means 17.
 */
fun wanTime(ctx: android.content.Context) =
    if (GpuEngine.useForWan(ctx)) WAN_GPU_TIME
    else if (java.io.File(ModelStore.WAN.modelDir(ctx), "frames.txt").takeIf { it.isFile }?.readText()?.trim() == "49") WAN_LONG_TIME else WAN_TIME
val SHAPES = listOf("Square", "Tall 9:16", "Wide 16:9")
private val SIZES = listOf(
    listOf(512 to 512, 384 to 672, 672 to 384),
    listOf(768 to 768, 576 to 1024, 1024 to 576),
)
fun pictureSize(tier: Int, shape: Int): Pair<Int, Int> = SIZES[tier][shape]

/** Klein's size tier: 768 on the graphics chip (512 drew broken bodies there) and with the 8 Elite Gen 5 AI-chip
 *  build (built for 768); 512 with the 8 Elite's AI-chip build. */
fun kleinTier(ctx: android.content.Context) = if (GpuEngine.useForKlein(ctx) || Chip.kind == Chip.Kind.ELITE_GEN5) 1 else 0

/** Klein's 8 Elite Gen 5 AI-chip build draws 768x768 only (three shapes in one piece would not pack). */
fun kleinSquareOnly(ctx: android.content.Context) = !GpuEngine.useForKlein(ctx) && Chip.kind == Chip.Kind.ELITE_GEN5

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

    // test builds only, when files/test-shortcut exists (made over the cable with run-as): `--es draw <model>
    // --es prompt "<text>" [--el seed N] [--ei steps N] [--ei tier N] [--es source <file in the queue folder>]` adds one
    // picture to the queue, to check every model from the computer
    private fun testDraw(intent: android.content.Intent) {
        if (!BuildConfig.DEBUG || !File(filesDir, "test-shortcut").exists()) return
        val model = intent.getStringExtra("draw") ?: return
        val id = java.util.UUID.randomUUID().toString().take(12)
        Queue.store.add(Job(id, System.currentTimeMillis(), model, "qwen", intent.getStringExtra("prompt") ?: "a cat",
            intent.getLongExtra("seed", 42L), intent.getIntExtra("steps", 8), intent.getIntExtra("tier", 0), 0,
            1f, 0.5f, intent.getStringExtra("source"), "create"))
        QueueService.kick(this)
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        testDraw(intent)
        intent.getStringExtra("openUri")?.let { openUri.value = it }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = getExternalFilesDir(null)!!
        val modelDir = File(root, "models")
        // The app must create these folders itself; one made over the cable first is not readable by the app.
        modelDir.mkdirs()
        ModelStore.ALL.forEach { st ->
            st.moveOldGpuInstall(this)
            st.modelDir(this).mkdirs(); st.gpu?.modelDir(this)?.mkdirs()
        }
        // the old Settings switch "Draw MOSS and Fibo with the graphics chip" becomes each model's own choice
        getSharedPreferences("anima", MODE_PRIVATE).let { p ->
            // only the old on/off value (a test build briefly kept the engine choice, a number, under this name)
            when (val old = p.all["gpuEngine"]) {
                null -> {}
                is Boolean -> {
                    if (old) listOf(ModelStore.MOSS, ModelStore.FIBO).forEach { it.setRunsOnGpu(this, true) }
                    p.edit().remove("gpuEngine").apply()
                }
                else -> p.edit().remove("gpuEngine").apply()
            }
        }
        val libDir = applicationInfo.nativeLibraryDir
        val selfTest = intent.getBooleanExtra("selftest", false)
        val loadOpts = intent.getIntExtra("loadopts", -1)
        val memOverride = intent.getIntExtra("memmode", -1)
        if (BuildConfig.DEBUG && File(filesDir, "test-unsupported").exists()) Chip.testOverride = Chip.Kind.UNSUPPORTED
        Engines.init(this, if (memOverride >= 0) memOverride else if (selfTest) 1 else -1, loadOpts)
        Queue.init(this)
        openUri.value = intent.getStringExtra("openUri")
        // notifications are a permission from Android 13; saving to Pictures/Anima needs storage access up to Android 9
        val ask = buildList {
            if (android.os.Build.VERSION.SDK_INT >= 33) add(android.Manifest.permission.POST_NOTIFICATIONS)
            if (android.os.Build.VERSION.SDK_INT < 29) {   // both: Android 9 opens shared storage only with read granted too
                add(android.Manifest.permission.READ_EXTERNAL_STORAGE); add(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }.filter { checkSelfPermission(it) != android.content.pm.PackageManager.PERMISSION_GRANTED }
        if (ask.isNotEmpty()) requestPermissions(ask.toTypedArray(), 1)
        // jobs left waiting (or interrupted) when the app was last closed
        if (!selfTest && Queue.store.waitingCount() > 0) QueueService.kick(this)
        // test builds: `am start ... --es chat "<message>" [--ez think true] [--es conv <id>] [--es image <file in
        // the app's files folder>]` sends a chat message. Only when files/test-shortcut exists (made over the cable
        // with run-as): published builds are debug builds too, and other apps can start this screen.
        if (BuildConfig.DEBUG && File(filesDir, "test-shortcut").exists()) intent.getStringExtra("chat")?.let { msg ->
            val img = intent.getStringExtra("image")?.let { android.graphics.BitmapFactory.decodeFile(File(filesDir, it).absolutePath) }
            Chat.get(this).send(intent.getStringExtra("conv"), msg, img, intent.getBooleanExtra("think", false))
        }
        testDraw(intent)

        Look.load(this)
        setContent {
            PaperTheme(Look.isBlack(Look.mode, androidx.compose.foundation.isSystemInDarkTheme())) {
                val ctx = this@MainActivity
                // installed variant per model (ModelStore id -> variant id or null)
                // (judged by the build the model draws with: AI chip or graphics chip, as chosen on the models page)
                fun scanInstalled() = ModelStore.ALL.associate { it.id to it.active(ctx).installed(ctx) }
                var installedOf by remember { mutableStateOf(scanInstalled()) }
                val installed = installedOf["anima"]
                var downloadFor by remember { mutableStateOf<String?>(null) }  // ModelStore id whose download screen is open
                var updating by remember { mutableStateOf(false) }
                var updateBytesOf by remember { mutableStateOf(mapOf<String, Long>()) }
                // Quietly look for newer model files once per launch (the lists are tiny).
                LaunchedEffect(installedOf) {
                    if (!selfTest) for (st in ModelStore.ALL) if (installedOf[st.id] != null) runCatching {
                        val a = st.active(ctx)
                        val m = a.fetchManifest(ctx)
                        updateBytesOf = updateBytesOf + (st.id to a.outdated(ctx, m).sumOf { it.size })
                    }
                }
                // a finished download, wherever the user is: the engine reads the new files (sizes, tables) afresh
                // (several can finish at once now that downloads run side by side)
                LaunchedEffect(Installer.finished.size) {
                    while (Installer.finished.isNotEmpty()) {
                        val sid = ModelStore.byId(Installer.finished.removeAt(0).first).baseId   // a graphics-chip build counts as its model
                        installedOf = scanInstalled()
                        if (downloadFor == sid) { downloadFor = null; updating = false }
                        updateBytesOf = updateBytesOf + (sid to 0L)
                        Engines.invalidate(sid)
                    }
                }
                val prefs = remember { getSharedPreferences("anima", MODE_PRIVATE) }
                // the first-run setup (choose models), until it is finished or skipped; phones that already had a
                // model when this screen arrived never see it, and Settings opens it again
                var showSetup by remember {
                    mutableStateOf(!selfTest && !prefs.getBoolean("setupDone", false) && ModelStore.ALL.all { installedOf[it.id] == null })
                }
                val create = remember {
                    CreateState().also {
                        it.memoryMode = prefs.getInt("memoryMode", 1)
                        it.tier = prefs.getInt("tier", 0).coerceIn(0, SIZE_TIERS.size - 1)
                        it.shape = prefs.getInt("shape", 0).coerceIn(0, SHAPES.size - 1)
                        it.model = (prefs.getString("model", "anima") ?: "anima")
                            .takeIf { m -> ModelStore.ALL.any { st -> st.id == m && st.offered } } ?: "anima"
                        it.qwenDrawing = prefs.getString("qwenDrawing", "qwen").takeIf { d -> d == "noctq" } ?: "qwen"
                        // each model's last prompt, as the user left it (GitHub issue 1: it went back to the example)
                        it.restorePrompts { m -> prefs.getString("prompt_$m", null) }
                    }
                }
                // save the prompt as it is typed (a moment after the last key), so leaving the app keeps it
                LaunchedEffect(Unit) {
                    snapshotFlow { create.model to create.prompt }.debounce(600).collect { (m, p) ->
                        prefs.edit().putString("prompt_$m", p).apply()
                    }
                }
                var tab by remember { mutableIntStateOf(0) }
                var open by remember { mutableStateOf<Gallery.Entry?>(null) }
                var showQueue by remember { mutableStateOf(false) }
                var galleryRefresh by remember { mutableIntStateOf(0) }   // reloads the grid after the side panel changed a picture
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
                fun go(t: Int) { showSetup = false; downloadFor = null; updating = false; showQueue = false; open = null; tab = t; navScope.launch { drawer.close() } }
                val wide = isWide()
                val paper = LocalPaper.current
                val menu: @Composable ColumnScope.() -> Unit = {
                        Row(Modifier.padding(start = 24.dp, top = 20.dp, bottom = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                            AppMark(); Spacer(Modifier.width(12.dp))
                            Text("Anima", style = MaterialTheme.typography.titleLarge)
                        }
                        val qTick by Queue.store.changed.collectAsState()
                        val waiting = remember(qTick) { Queue.store.waitingCount() }
                        @Composable fun item(label: String, icon: androidx.compose.ui.graphics.vector.ImageVector, selected: Boolean,
                                             badge: Int = 0, onClick: () -> Unit) =
                            NavigationDrawerItem(label = { Text(label, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium) },
                                selected = selected, onClick = onClick,
                                icon = { Icon(icon, null) }, badge = { if (badge > 0) CountBadge(badge) },
                                colors = NavigationDrawerItemDefaults.colors(
                                    selectedContainerColor = paper.soft, unselectedContainerColor = Color.Transparent,
                                    selectedIconColor = paper.softText, unselectedIconColor = paper.muted,
                                    selectedTextColor = paper.text, unselectedTextColor = paper.muted),
                                modifier = Modifier.padding(horizontal = 12.dp))
                        val here = downloadFor == null && open == null && !showQueue
                        item("Create", AppIcons.Create, here && tab == 0) { go(0) }
                        item("Chat", AppIcons.Chat, here && tab == 1) { go(1) }
                        item("Gallery", AppIcons.Gallery, here && tab == 2) { go(2) }
                        item("Queue", AppIcons.Queue, showQueue, badge = waiting) { go(0); showQueue = true }
                        item("Settings", AppIcons.Settings, here && tab == 3) { go(3) }
                }
                val useEntry: (Gallery.Entry) -> Unit = { e ->
                    create.prompt = e.prompt; create.seed = e.seed; if (create.model == "anima") create.steps = e.steps
                    sizeIndex(e.width, e.height)?.let { (t, k) -> create.tier = t; create.shape = k }
                    open = null; tab = 0
                }
                val body: @Composable () -> Unit = {
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
                            showSetup && !prefs.getBoolean("setupDone", false) -> Readable { SetupScreen(firstRun = true, onDone = {
                                prefs.edit().putBoolean("setupDone", true).apply()
                                showSetup = false; tab = 0
                            }) }
                            // the models page: download, delete, update, and which chip each model draws on
                            showSetup -> Readable { ModelsScreen(onChanged = { installedOf = scanInstalled() },
                                onBack = { installedOf = scanInstalled(); showSetup = false }) }
                            downloadFor != null -> {
                                val st = ModelStore.byId(downloadFor ?: "anima").active(ctx)
                                Readable { DownloadScreen(
                                    store = st,
                                    current = st.installed(ctx),
                                    beforeSwap = { Engine.release() },
                                    onBack = { downloadFor = null; updating = false },
                                    update = updating,
                                ) }
                            }
                            showQueue && open == null -> Readable { QueueScreen(onOpen = { open = it }, onBack = { showQueue = false }) }
                            // unfolded, a picture opened from the gallery shows in a panel beside the grid (below)
                            open != null && !(isWide() && tab == 2 && !showQueue) -> Detail(open!!, onBack = { open = null; galleryRefresh++ }, onUse = useEntry)
                            tab == 0 -> Column {
                                val qTick by Queue.store.changed.collectAsState()
                                val waitingNow = remember(qTick) { Queue.store.waitingCount() }
                                ChatBar(openMenu, "Create") { QueueChip(waitingNow, Modifier.padding(end = 12.dp)) { showQueue = true } }
                                val st = ModelStore.byId(create.model)
                                val inst = installedOf[st.id]
                                Create(create, Engines::ensureAnima, Engines::ensureKlein, { Engines.ensureQwen(create.qwenDrawing) },
                                    modelLabel = st.title + (st.active(ctx).cachedManifest(ctx)?.variants?.firstOrNull { it.id == inst }?.label?.takeIf { it != st.title }?.let { " · $it" } ?: ""),
                                    installed = inst != null, animaInstalled = installed != null,
                                    onChangeModel = { downloadFor = st.id }, onDownload = { downloadFor = st.id },
                                    updateBytes = updateBytesOf[st.id] ?: 0L, onUpdate = { updating = true; downloadFor = st.id },
                                    onModel = { id, drawing ->
                                        if (id != create.model) Engine.release()  // the other model's pieces leave the neural unit
                                        create.model = id
                                        if (drawing != null) create.qwenDrawing = drawing
                                        prefs.edit().putString("model", id).putString("qwenDrawing", create.qwenDrawing).apply()
                                    },
                                    onSize = { prefs.edit().putInt("tier", create.tier).putInt("shape", create.shape).apply() },
                                    onQueue = { showQueue = true }, onOpen = { open = it }, onSeeAll = { go(2) }) { m ->
                                    create.memoryMode = m
                                    prefs.edit().putInt("memoryMode", m).apply()
                                    Engines.setMemoryMode(m)
                                }
                            }
                            tab == 1 -> Box(Modifier.fillMaxSize()) { ChatScreen(Chat.get(ctx), isInstalled = { installedOf[it] != null },
                                onDownload = { downloadFor = LocalLing.store(ctx).id }, onOpen = { open = it }, onMenu = openMenu) }
                            tab == 3 -> Readable { Column { ChatBar(openMenu, "Settings"); SettingsScreen(onSetup = { showSetup = true }) } }
                            else -> if (isWide() && open != null) Row(Modifier.fillMaxSize()) {
                                Box(Modifier.weight(1f)) { GalleryGrid(openMenu, onOpen = { open = it }, selected = open, refresh = galleryRefresh) }
                                Box(Modifier.width(300.dp).fillMaxHeight()) {
                                    Detail(open!!, onBack = { open = null; galleryRefresh++ }, onUse = useEntry, panel = true)
                                }
                            } else GalleryGrid(openMenu, onOpen = { open = it }, refresh = galleryRefresh)
                        }
                    }
                }
                }
                // wide (unfolded): a slim strip of the pages down the side, always there (the open menu, 232 dp, left the
                // Create settings too cramped on a Fold's inner screen); narrow: the menu slides out from its button
                if (wide) Row(Modifier.fillMaxSize()) {
                    val qTick by Queue.store.changed.collectAsState()
                    val waiting = remember(qTick) { Queue.store.waitingCount() }
                    val here = downloadFor == null && open == null && !showQueue
                    NavigationRail(Modifier.safeDrawingPadding(), containerColor = paper.background,
                        header = { Spacer(Modifier.height(10.dp)); AppMark(); Spacer(Modifier.height(10.dp)) }) {
                        @Composable fun rail(label: String, icon: androidx.compose.ui.graphics.vector.ImageVector, selected: Boolean,
                                             badge: Int = 0, onClick: () -> Unit) =
                            NavigationRailItem(selected = selected, onClick = onClick,
                                label = { Text(label, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium) },
                                icon = { if (badge > 0) BadgedBox(badge = { CountBadge(badge) }) { Icon(icon, label) } else Icon(icon, label) },
                                colors = NavigationRailItemDefaults.colors(indicatorColor = paper.soft,
                                    selectedIconColor = paper.softText, unselectedIconColor = paper.muted,
                                    selectedTextColor = paper.text, unselectedTextColor = paper.muted))
                        Spacer(Modifier.height(8.dp))
                        rail("Create", AppIcons.Create, here && tab == 0) { go(0) }
                        rail("Chat", AppIcons.Chat, here && tab == 1) { go(1) }
                        rail("Gallery", AppIcons.Gallery, here && tab == 2) { go(2) }
                        rail("Queue", AppIcons.Queue, showQueue, badge = waiting) { go(0); showQueue = true }
                        rail("Settings", AppIcons.Settings, here && tab == 3) { go(3) }
                    }
                    VerticalDivider(color = paper.divider)
                    Box(Modifier.weight(1f)) { body() }
                }
                else ModalNavigationDrawer(drawerState = drawer, gesturesEnabled = open == null, drawerContent = {
                    ModalDrawerSheet(Modifier.safeDrawingPadding(), drawerContainerColor = paper.background) { menu() }
                }) { body() }
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
    updateBytes: Long, onUpdate: () -> Unit, onModel: (String, String?) -> Unit, onSize: () -> Unit,
    onQueue: () -> Unit, onOpen: (Gallery.Entry) -> Unit = {}, onSeeAll: () -> Unit = {}, onMemoryMode: (Int) -> Unit,
) {
    val ctx = LocalContext.current
    val scope = s.work  // outlives this screen: a tab switch must not drop a picture being drawn
    val klein = s.model == "klein"
    val qwen = s.model == "qwen"
    val fibo = s.model == "fibo"
    val moss = s.model == "moss"
    val wan = s.model == "wan"
    val big = s.model in BIG_MODELS   // one 768x768 size, no size or memory choices
    val anima = s.model == "anima"
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
    if (ModelStore.PICTURE.none { it.offered }) {   // phones without the AI chip still get MOSS on the graphics chip
        DeviceNotSupported(Modifier.padding(16.dp))
        return
    }
    val controls: @Composable ColumnScope.() -> Unit = controls@{
        // One list of drawing models (2026-11-06: five models and Noct Q no longer fit as buttons in a row).
        // Noct Q is Qwen with other drawing files, copied over the cable into qwen/noctq/: listed when present.
        val choices = remember(installed) {
            listOfNotNull(
                Triple("anima", null, "Anima"), Triple("klein", null, "FLUX.2 klein"), Triple("qwen", "qwen", "Qwen-Image 2.1"),
                Triple("qwen", "noctq", "Noct Q").takeIf { File(ModelStore.QWEN.modelDir(ctx), "$NOCTQ_DIR/sp31.bin").isFile },
                Triple("fibo", null, "Fibo 2.0 (edits)"), Triple("moss", null, "MOSS-Image"),
                Triple("wan", null, "Wan 2.2 (moving picture)"),
            )   // every model is listed; one with nothing for this phone is shown as unavailable
        }
        var menu by remember { mutableStateOf(false) }
        val p = LocalPaper.current
        val modelPicker: @Composable () -> Unit = {
            Box {
                Pill((choices.firstOrNull { it.first == s.model && (it.second == null || it.second == s.qwenDrawing) }?.third ?: s.model) + "  ▾",
                    enabled = !s.busy) { menu = true }
                DropdownMenu(menu, { menu = false }) {
                    choices.forEach { (id, drawing, name) ->
                        val st = ModelStore.byId(id)
                        val have = st.active(ctx).installed(ctx) != null   // the version this phone uses (AI chip or graphics chip)
                        DropdownMenuItem(text = { Text(when { !st.offered -> "$name · unavailable on this phone"; have -> name; else -> "$name · not downloaded" }) },
                            enabled = st.offered, onClick = { menu = false; onModel(id, drawing) })
                    }
                }
            }
        }
        @Composable fun Note(text: String, color: Color = p.muted) = Text(text, style = MaterialTheme.typography.bodySmall, color = color)
        if (!installed) {
            PaperCard(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Model", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f)); modelPicker()
                    }
                    Text("${ModelStore.byId(s.model).title} is not downloaded yet." + when {
                        klein -> " It draws from plain sentences and can edit your pictures (about 4 GB)."
                        qwen -> " It draws 768×768 pictures from plain sentences and can edit your pictures (about 15 GB)."
                        fibo -> " It changes your pictures as you describe, at 768×768 (about 15 GB). It only edits."
                        moss -> " It draws 768×768 pictures from plain sentences in 8 steps (about ${if (ModelStore.MOSS.runsOnGpu(ctx)) 11 else 16} GB). It does not edit."
                        wan -> " It turns one of your pictures into a short looping video that moves as you describe."
                        else -> ""
                    })
                    val dl = ModelStore.byId(s.model).active(ctx).id
                    val task = Installer.task(dl)
                    BigButton("Download…", enabled = !Installer.isRunning(dl), modifier = Modifier.fillMaxWidth(), onClick = onDownload)
                    if (task != null && task.running) {
                        Note("Downloading: ${task.status}")
                        LinearProgressIndicator(progress = { task.fraction }, modifier = Modifier.fillMaxWidth())
                    }
                }
            }
            return@controls
        }
        val fiboNeedsPicture = (fibo || wan) && s.edit == null

        // the prompt card: what to draw, with which model, and any picture to edit or follow
        PaperCard(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(if (wan) "How the picture moves" else if (fibo || !anima && !moss && !qwenSmall && s.edit != null) "What to change" else "Prompt",
                        fontSize = 15.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                    modelPicker()
                }
                BasicTextField(s.prompt, { s.prompt = it }, Modifier.fillMaxWidth().heightIn(min = 96.dp), enabled = !s.busy,
                    textStyle = androidx.compose.ui.text.TextStyle(fontSize = 17.sp, lineHeight = 26.sp, color = p.text, fontFamily = PaperFont),
                    cursorBrush = androidx.compose.ui.graphics.SolidColor(p.accent))
                if (qwenSmall) Note("Editing pictures needs the Standard Qwen download (Model: Change).")
                if (fiboNeedsPicture) Note(if (wan) "Choose the picture to bring to life below." else "Fibo only edits: choose a picture to edit below.")
                if (!anima && !qwenSmall && !moss) {   // MOSS only draws new pictures
                    Note(if (wan) "Picture to animate: it becomes the first frame" else if (fibo) "Picture to edit: Fibo changes it as you describe"
                        else "Picture to edit (optional): ${if (klein) "Klein" else "Qwen"} changes it as you describe")
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        s.edit?.let { Image(it.asImageBitmap(), null, Modifier.size(64.dp).clip(RoundedCornerShape(10.dp)), contentScale = ContentScale.Crop) }
                        DashedButton(if (s.edit == null) "Choose picture" else "Change", AppIcons.Gallery, enabled = !s.busy) {
                            pickEdit.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
                        if (s.edit != null) TextButton(onClick = { s.edit = null }, enabled = !s.busy) { Text("Remove", maxLines = 1) }
                    }
                    if (s.edit != null) Note((if (big) "It is cropped to a square." else "It is cropped to the shape below.") +
                        if (wan) " Describe the movement, e.g. \"she waves and laughs\"." else " Describe the change, e.g. \"make her hair red\".")
                }
                if (anima) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        s.guide?.let { Image(it.asImageBitmap(), null, Modifier.size(64.dp).clip(RoundedCornerShape(10.dp)), contentScale = ContentScale.Crop) }
                        DashedButton(if (s.guide == null) "Guide picture" else "Change", AppIcons.Gallery, enabled = !s.busy) {
                            pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
                        if (s.guide != null) TextButton(onClick = { s.guide = null }, enabled = !s.busy) { Text("Remove", maxLines = 1) }
                    }
                    if (s.guide == null) Note("Optional: the layout and pose follow it, the prompt decides the rest.")
                    else {
                        Note("How closely to follow it: %.1f".format(s.guideStrength))
                        Slider(s.guideStrength, { s.guideStrength = (it * 10).toInt() / 10f }, valueRange = 0.2f..1.5f, enabled = !s.busy)
                        Note("For how much of the drawing: %d%%".format((s.guideEnd * 100).toInt()))
                        Slider(s.guideEnd, { s.guideEnd = (it * 20).toInt() / 20f }, valueRange = 0.2f..1f, enabled = !s.busy)
                        Note("Less of the drawing keeps the pose but lets the prompt change clothes and details.")
                    }
                }
            }
        }

        // the settings card: shape and size up front, the rarely changed ones behind one row
        val squareOnly = klein && kleinSquareOnly(ctx)   // Klein's 8 Elite Gen 5 AI-chip build: other shapes greyed out
        val shapeUsed = if (squareOnly) 0 else s.shape
        val (pw, ph) = if (big) 768 to 768 else pictureSize(if (klein) kleinTier(ctx) else s.tier, shapeUsed)
        val time = when { klein -> if (GpuEngine.useForKlein(ctx)) KLEIN_GPU_TIME else if (Chip.kind == Chip.Kind.ELITE_GEN5) KLEIN_GEN5_TIME else KLEIN_TIME
            fibo -> if (GpuEngine.useForFibo(ctx)) FIBO_GPU_TIME else FIBO_TIME
            moss -> if (GpuEngine.useForMoss(ctx)) MOSS_GPU_TIME else MOSS_TIME
            wan -> wanTime(ctx)
            qwen -> if (s.qwenDrawing == "noctq") NOCTQ_TIME else QWEN_TIME; else -> SIZE_TIERS[s.tier].second }
        val myTask = Installer.task(ModelStore.byId(s.model).active(ctx).id)?.takeIf { it.running }
        PaperCard(Modifier.fillMaxWidth(), padding = PaddingValues(horizontal = 16.dp)) {
            if (!big) Column(Modifier.padding(vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Shape", fontSize = 15.sp, fontWeight = FontWeight.Medium)
                PillChoices(SHAPES, shapeUsed, enabled = !s.busy, enabledAt = { !squareOnly || it == 0 }) { s.shape = it; onSize() }
                if (squareOnly) Note("Tall and wide are not built for this phone's AI chip yet. The graphics chip draws them.")
            }
            if (!big) HorizontalDivider(color = p.divider)
            Column(Modifier.padding(vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (anima) {
                    Text("Size", fontSize = 15.sp, fontWeight = FontWeight.Medium)
                    PillChoices(SIZE_TIERS.map { it.first }, s.tier, enabled = !s.busy) { s.tier = it; onSize() }
                }
                Note(listOf("$pw×$ph", time).filter { it.isNotEmpty() }.joinToString(" · "))
            }
            HorizontalDivider(color = p.divider)
            PaperRow(Modifier.clickable { s.advanced = !s.advanced }, divider = s.advanced) {
                Text("Steps, seed and memory", Modifier.weight(1f), fontSize = 15.sp, fontWeight = FontWeight.Medium)
                Text(if (anima) "${s.steps} steps" else "seed ${s.seed}", fontSize = 13.sp, color = p.muted, maxLines = 1)
                Icon(AppIcons.Chevron, null, Modifier.rotate(if (s.advanced) 90f else 0f), tint = p.muted)
            }
            if (s.advanced) Column(Modifier.padding(vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        s.seed.toString(), { v -> v.toLongOrNull()?.let { s.seed = it; s.seedLocked = true } }, Modifier.width(170.dp),
                        label = { Text("Seed") }, singleLine = true, enabled = !s.busy,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    )
                    Pill("New seed", enabled = !s.busy) { s.seed = Random.nextLong(0, 1_000_000_000); s.seedLocked = false }
                }
                if (anima) {
                    Text("Steps (fewer is faster, 8 is the full quality)", fontSize = 15.sp, fontWeight = FontWeight.Medium)
                    PillChoices(listOf("4", "6", "8"), listOf(4, 6, 8).indexOf(s.steps), enabled = !s.busy) { s.steps = listOf(4, 6, 8)[it] }
                }
                if (!big) {
                    Text("Memory use", fontSize = 15.sp, fontWeight = FontWeight.Medium)
                    PillChoices(MEMORY_MODES.map { it.first }, s.memoryMode, enabled = !s.busy) { onMemoryMode(it) }
                    Note(MEMORY_MODES[s.memoryMode].second)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Note("Model: $modelLabel", ); Spacer(Modifier.weight(1f))
                    TextButton(onClick = onChangeModel, enabled = !s.busy) { Text("Change") }
                }
            }
        }
        if (myTask != null) {
            Note("Updating the model: ${myTask.status}")
            LinearProgressIndicator(progress = { myTask.fraction }, modifier = Modifier.fillMaxWidth())
        } else if (updateBytes > 0) Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Model update available (%.1f GB)".format(updateBytes / 1e9), style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            TextButton(onClick = onUpdate, enabled = !s.busy) { Text("Update") }
        }
    }
    // Draw: one tap, one job. The button shows a spinner until the job is in the queue, so a second tap (or a
    // double tap) cannot add it twice.
    val drawButton: @Composable () -> Unit = {
        val myTask = Installer.task(ModelStore.byId(s.model).active(ctx).id)?.takeIf { it.running }
        val fiboNeedsPicture = (fibo || wan) && s.edit == null
        var adding by remember { mutableStateOf(false) }
        val addScope = rememberCoroutineScope()
        BigButton(if (adding) "Adding…" else if (myTask != null) "Draw (after the update)" else "Draw",
            icon = AppIcons.Brush, busy = adding,
            enabled = !adding && myTask == null && s.prompt.isNotBlank() && installed && !fiboNeedsPicture,
            modifier = Modifier.fillMaxWidth(), onClick = {
            if (adding) return@BigButton
            adding = true
            addScope.launch {
                try {
                    val t0 = System.currentTimeMillis()
                    val j = withContext(Dispatchers.IO) { enqueueFromCreate(ctx, s) }
                    if (Queue.store.get(j.id) != null) {
                        if (!s.seedLocked) s.seed = Random.nextLong(0, 1_000_000_000)   // the next job draws a new picture
                        s.status = "Added to the queue"
                    }
                    kotlinx.coroutines.delay((500 - (System.currentTimeMillis() - t0)).coerceAtLeast(0))
                } catch (e: Exception) {
                    s.status = "Could not add to the queue: ${e.message}"
                } finally { adding = false }
            }
        })
    }
    val result: @Composable ColumnScope.() -> Unit = {
        // the newest finished picture from this screen, and how the queue is doing
        val p = LocalPaper.current
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
            val bmp = withContext(Dispatchers.IO) { Gallery.full(ctx, e, Int.MAX_VALUE) } ?: return@LaunchedEffect
            s.shownJob = j.id
            s.picture = bmp; s.savedFor = bmp; s.genW = bmp.width; s.pictureMs = j.ms
            s.clipUri = if (j.model == "wan") e.uri else null
            s.status = "Done in %.0f s · saved to the gallery".format(j.ms / 1000.0)
        }
        if (queueLine.isNotEmpty()) TextButton(onClick = onQueue) { Text(queueLine, style = MaterialTheme.typography.bodySmall) }
        val bmp = s.picture
        if (bmp == null) { if (s.status.isNotEmpty()) Text(s.status, style = MaterialTheme.typography.bodySmall, color = p.muted) }
        else PaperCard(Modifier.fillMaxWidth(), padding = PaddingValues(12.dp)) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                val clip = s.clipUri
                val m = Modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height).clip(RoundedCornerShape(14.dp))
                if (clip != null) GifPlayer(clip, m) else Image(bmp.asImageBitmap(), null, m)
                if (s.status.isNotEmpty()) Text(s.status, Modifier.padding(horizontal = 4.dp), style = MaterialTheme.typography.bodySmall, color = p.muted)
                if (clip == null) Row(Modifier.padding(horizontal = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Pill(if (s.savedFor === bmp) "Saved" else "Save", enabled = !s.busy && s.savedFor !== bmp) {
                        scope.launch(Dispatchers.IO) {
                            Gallery.save(ctx, bmp, s.prompt, s.seed, if (anima) s.steps else 4, s.pictureMs)
                            s.savedFor = bmp
                            s.status = "Saved to Pictures/Anima (${bmp.width}×${bmp.height})"
                        }
                    }
                    if (bmp.width == s.genW && animaInstalled) Pill("Upscale 4×", enabled = !s.busy && !Installer.isRunning("anima")) {
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
                    }
                    // the finished picture as the next drawing's guide: the bitmap already in memory, no file written
                    if (anima) Pill("Use as guide", enabled = !s.busy && s.guide !== bmp) { s.guide = bmp; s.status = "Set as the guide picture" }
                }
            }
        }
    }
    if (isWide()) Row(Modifier.fillMaxSize()) {
        Column(Modifier.width(300.dp).fillMaxHeight().padding(start = 16.dp, top = 8.dp, bottom = 16.dp)) {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                controls(); Spacer(Modifier.height(8.dp))
            }
            if (installed) { Spacer(Modifier.height(12.dp)); drawButton() }
        }
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            if (installed) result()
            if (s.picture == null) Text("The finished picture shows here.", style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            RecentStrip(5, onOpen, onSeeAll)
            Spacer(Modifier.height(24.dp))
        }
    } else Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        controls()
        RecentStrip(4, onOpen, onSeeAll)
        if (installed) drawButton()
        if (installed) result()
        Spacer(Modifier.height(24.dp))
    }
}

/** The newest pictures, [columns] across, with "See all" for the gallery. Reloads when the queue finishes one. */
@Composable
private fun RecentStrip(columns: Int, onOpen: (Gallery.Entry) -> Unit, onSeeAll: () -> Unit) {
    val ctx = LocalContext.current
    val p = LocalPaper.current
    val tick by Queue.store.changed.collectAsState()
    val recent by produceState(emptyList<Pair<Gallery.Entry, Bitmap?>>(), tick, columns) {
        value = withContext(Dispatchers.IO) { Gallery.list(ctx).take(columns).map { it to Gallery.thumbnail(ctx, it, 256) } }
    }
    if (recent.isEmpty()) return
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(Modifier.padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Recent", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            Text("See all", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = p.link,
                modifier = Modifier.clickable(onClick = onSeeAll).padding(4.dp))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (i in 0 until columns) {
                val item = recent.getOrNull(i)
                Box(Modifier.weight(1f).aspectRatio(1f).clip(RoundedCornerShape(14.dp)).background(if (item == null) Color.Transparent else p.card)
                    .then(if (item != null) Modifier.clickable { onOpen(item.first) } else Modifier)) {
                    item?.second?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
                }
            }
        }
    }
}

/**
 * The saved pictures. Press and hold one to start choosing (2026-11-18): taps then tick and untick pictures, the bar
 * shows how many are chosen with "choose all" and delete, and Back or the cross stops choosing.
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun GalleryGrid(openMenu: () -> Unit, onOpen: (Gallery.Entry) -> Unit, selected: Gallery.Entry? = null, refresh: Int = 0) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var items by remember { mutableStateOf<List<Gallery.Entry>>(emptyList()) }
    var loaded by remember { mutableStateOf(false) }
    var chosen by remember { mutableStateOf<Set<String>>(emptySet()) }
    var choosing by remember { mutableStateOf(false) }
    var asking by remember { mutableStateOf(false) }
    var clips by remember { mutableStateOf<Set<String>>(emptySet()) }   // which pictures are GIFs or videos
    LaunchedEffect(refresh) {
        // the pictures show at once; which are clips is asked of the phone one by one afterwards (slow on big galleries)
        val all = withContext(Dispatchers.IO) { Gallery.list(ctx) }
        items = all
        loaded = true
        clips = withContext(Dispatchers.IO) { all.filter { Gallery.isClip(ctx, it) }.map { it.uri.toString() }.toSet() }
    }
    val shown = remember(items, clips, galleryFilter) { items.filter { galleryFilter.matches(it.kind, it.uri.toString() in clips) } }
    val groups = remember(shown) { groupByDay(shown, { it.savedAt }, System.currentTimeMillis(), java.time.ZoneId.systemDefault()) }
    fun stop() { choosing = false; chosen = emptySet() }
    BackHandler(enabled = choosing) { stop() }
    val scheme = MaterialTheme.colorScheme
    Column(Modifier.fillMaxSize()) {
        if (choosing) Row(Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = { stop() }) { Icon(AppIcons.Close, "Stop choosing") }
            Text(if (chosen.isEmpty()) "Choose pictures" else "${chosen.size} chosen", Modifier.weight(1f).padding(start = 8.dp),
                fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
            IconButton(onClick = {
                chosen = if (chosen.size == shown.size) emptySet() else shown.map { it.uri.toString() }.toSet()
            }) { Icon(AppIcons.SelectAll, if (chosen.size == shown.size) "Choose none" else "Choose all") }
            IconButton(onClick = { asking = true }, enabled = chosen.isNotEmpty()) { Icon(AppIcons.Delete, "Delete chosen") }
        } else ChatBar(openMenu, "Gallery", buttons = 1) {
            Pill("Select", enabled = shown.isNotEmpty(), modifier = Modifier.padding(end = 12.dp)) { choosing = true }
        }
        if (loaded && items.isNotEmpty()) PillChoices(GalleryFilter.values().map { it.label }, galleryFilter.ordinal,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)) { galleryFilter = GalleryFilter.values()[it] }
        if (asking) AlertDialog(onDismissRequest = { asking = false },
            title = { Text(if (chosen.size == 1) "Delete this picture?" else "Delete ${chosen.size} pictures?") },
            text = { Text("They are removed from the phone, not only from this list.") },
            confirmButton = { TextButton(onClick = {
                asking = false
                val gone = items.filter { it.uri.toString() in chosen }
                scope.launch {
                    withContext(Dispatchers.IO) { Gallery.deleteMany(ctx, gone) }
                    items = items - gone.toSet()
                    stop()
                }
            }) { Text("Delete") } },
            dismissButton = { TextButton(onClick = { asking = false }) { Text("Cancel") } })
        if (loaded && shown.isEmpty()) {
            Text(when {
                items.isEmpty() -> "No saved pictures yet."
                galleryFilter == GalleryFilter.CLIPS -> "No clips yet."
                galleryFilter == GalleryFilter.EDITS -> "No edits yet. Pictures you change with Klein, Qwen or Fibo, and upscales, show here."
                else -> "No pictures yet."
            }, Modifier.padding(16.dp), color = LocalPaper.current.muted)
            return@Column
        }
        LazyVerticalGrid(GridCells.Fixed(if (isWide()) 4 else 3), Modifier.fillMaxSize().padding(horizontal = 13.dp, vertical = 4.dp)) {
          for ((day, dayItems) in groups) {
            item(key = "day:$day", span = { GridItemSpan(maxLineSpan) }) { SectionLabel(day, Modifier.padding(top = 10.dp, bottom = 4.dp)) }
            items(dayItems, key = { it.uri.toString() }) { e ->
                // each preview is made when its square comes into view, and kept for next time
                val thumb by produceState(Gallery.cachedThumbnail(e, 384), e.uri) {
                    if (value == null) value = withContext(Dispatchers.IO) { Gallery.thumbnail(ctx, e, 384) }
                }
                val key = e.uri.toString()
                val on = key in chosen
                fun toggle() { chosen = if (on) chosen - key else chosen + key }
                Box(Modifier.padding(3.dp).aspectRatio(1f)
                    .then(if (selected?.uri == e.uri) Modifier.border(2.dp, LocalPaper.current.accent, RoundedCornerShape(12.dp)).padding(4.dp) else Modifier)
                    .clip(RoundedCornerShape(if (selected?.uri == e.uri) 9.dp else 12.dp)).background(LocalPaper.current.card).combinedClickable(
                    onClick = { if (choosing) toggle() else onOpen(e) },
                    onLongClick = { if (!choosing) { choosing = true; chosen = setOf(key) } else toggle() })) {
                    thumb?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize().padding(if (on) 10.dp else 0.dp)
                        .clip(RoundedCornerShape(if (on) 8.dp else 0.dp)), contentScale = ContentScale.Crop) }
                    if (key in clips) Row(Modifier.align(Alignment.BottomStart).padding(6.dp).clip(CircleShape)
                        .background(Color.Black.copy(alpha = 0.72f)).padding(horizontal = 7.dp, vertical = 3.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        Text("▶", color = Color.White, fontSize = 9.sp)
                        Spacer(Modifier.width(3.dp))
                        Text("Clip", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.SemiBold)
                    }
                    if (choosing) Box(Modifier.align(Alignment.TopStart).padding(6.dp).size(24.dp).clip(CircleShape)
                        .background(if (on) scheme.surface else Color.Black.copy(alpha = 0.25f))) {
                        if (on) Icon(AppIcons.CheckCircle, null, Modifier.fillMaxSize(), tint = scheme.primary)
                        else Box(Modifier.fillMaxSize().border(2.dp, Color.White, CircleShape))
                    }
                }
            }
          }
        }
    }
}

/** The gallery's filter, kept while the app is open (leaving the page and coming back keeps it). */
private var galleryFilter by mutableStateOf(GalleryFilter.ALL)

@Composable
fun Detail(e0: Gallery.Entry, onBack: () -> Unit, onUse: (Gallery.Entry) -> Unit, panel: Boolean = false) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var e by remember(e0) { mutableStateOf(e0) }   // replaced by the upscaled picture's entry after Upscale
    var bmp by remember { mutableStateOf<Bitmap?>(null) }
    BackHandler(onBack = onBack)
    var gif by remember(e0) { mutableStateOf(false) }
    var note by remember(e0) { mutableStateOf("") }
    var upscaling by remember(e0) { mutableStateOf(false) }
    LaunchedEffect(e) { bmp = withContext(Dispatchers.IO) { gif = Gallery.isClip(ctx, e); Gallery.full(ctx, e) } }
    val picture: @Composable () -> Unit = {
        bmp?.let {
            val m = Modifier.fillMaxWidth().aspectRatio(it.width.toFloat() / it.height).clip(RoundedCornerShape(14.dp))
            if (gif) GifPlayer(e.uri, m) else Image(it.asImageBitmap(), null, m)
        }
    }
    val wide = isWide() && !panel   // as a side panel: one column, the picture on top
    Row(Modifier.fillMaxSize()) {
    if (wide) Column(Modifier.weight(1.3f).verticalScroll(rememberScrollState()).padding(16.dp)) {
        TextButton(onClick = onBack) { Text("‹ Back") }
        picture()
    }
    Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        if (!wide) { TextButton(onClick = onBack) { Text(if (panel) "Close" else "‹ Back") }; picture() } else Spacer(Modifier.height(40.dp))
        Text(e.prompt)
        Text("seed ${e.seed} · ${e.steps} steps · ${e.width}×${e.height}", style = MaterialTheme.typography.bodySmall)
        // wraps onto a second line when the column is too narrow for all three (the unfolded side panel)
        @OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
        androidx.compose.foundation.layout.FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Button(onClick = { onUse(e) }) { Text("Use this prompt") }
            Pill("Share", modifier = Modifier.height(40.dp)) {
                val send = Intent(Intent.ACTION_SEND).setType(Gallery.mime(ctx, e)).putExtra(Intent.EXTRA_STREAM, e.uri)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                ctx.startActivity(Intent.createChooser(send, "Share picture"))
            }
            Pill("Delete", enabled = !upscaling, modifier = Modifier.height(40.dp)) {
                scope.launch(Dispatchers.IO) { Gallery.delete(ctx, e); withContext(Dispatchers.Main) { onBack() } } }
        }
        // 4x bigger, in place of this picture. Up to 1024 on the long side: 4x of more would not fit in memory.
        val b = bmp
        // the upscaler runs on the AI chip: phones without one never get it, so no button that can only say "download Anima"
        if (Chip.supported && !gif && b != null && maxOf(b.width, b.height) <= 1024) {
            Pill("Upscale 4× (replaces this picture)", enabled = !upscaling && Upscaler.available(ctx)) {
                upscaling = true
                note = if (QueueService.progress.value.first.isNotEmpty()) "Upscaling after the picture being drawn…" else "Upscaling…"
                // not this screen's scope: leaving the screen must not drop the bigger picture half way
                Upscaler.scope.launch {
                    val t = System.currentTimeMillis()
                    val r = runCatching {
                        val big = Upscaler.upscale(ctx, b)
                        withContext(Dispatchers.IO) { Gallery.replace(ctx, e, big) } to big
                    }
                    withContext(Dispatchers.Main) {
                        r.onSuccess { (ne, big) ->
                            e = ne; bmp = big
                            note = "Upscaled to ${big.width}×${big.height} in %.0f s; it replaced the original".format((System.currentTimeMillis() - t) / 1000.0)
                        }.onFailure { note = "Could not upscale: ${it.message}" }
                        upscaling = false
                    }
                }
            }
            if (!Upscaler.available(ctx)) Text("The upscaler comes with Anima: download Anima to use it.", style = MaterialTheme.typography.bodySmall)
        }
        if (note.isNotEmpty()) Text(note, style = MaterialTheme.typography.bodySmall)
        // any picture as a home-screen icon: Android only lets the launcher icon itself be one of the built-in ones
        Pill("Add to home screen as an icon", enabled = bmp != null) {
            val b2 = bmp ?: return@Pill
            if (!AppIcon.pinPicture(ctx, b2, e.seed.toString() + "-" + e.uri.lastPathSegment))
                android.widget.Toast.makeText(ctx, "This launcher cannot add shortcuts", android.widget.Toast.LENGTH_SHORT).show()
        }
    }
    }
}

/** The app's mark in the menu: a small blue square with "A". */
@Composable
private fun AppMark() = Box(Modifier.size(40.dp).clip(RoundedCornerShape(12.dp)).background(LocalPaper.current.accent),
    contentAlignment = Alignment.Center) {
    Text("A", color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Bold)
}

/** How many pictures wait in the queue, as a small blue badge. */
@Composable
private fun CountBadge(n: Int) = Badge(containerColor = LocalPaper.current.badge, contentColor = Color.White) { Text("$n", style = BadgeNumber.copy(fontSize = 11.sp, lineHeight = 11.sp)) }
