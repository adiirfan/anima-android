package com.anima.npu

import android.app.Activity
import android.app.Instrumentation
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.intent.Intents
import androidx.test.espresso.intent.Intents.intending
import androidx.test.espresso.intent.matcher.IntentMatchers.isInternal
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.BySelector
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import org.hamcrest.Matchers.not
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File

/**
 * The on-phone release check: it opens the real app and taps through it the way a person would, with whatever models
 * this phone has downloaded. tools/release-check.sh runs it; tools/publish-app.sh refuses to publish without a pass.
 *
 * Everything it adds (pictures, queue jobs, chats, settings) is removed again at the end, pass or fail. The photo
 * picker is never opened: the app is handed one of the check's own pictures, so no private photo shows on screen.
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class ReleaseCheck {
    companion object {
        private const val TAG = "ReleaseCheck"
        private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
        private val device: UiDevice get() = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        private lateinit var scenario: ActivityScenario<MainActivity>

        // what was there before, so the check removes exactly what it added
        private var prefsBefore: Map<String, *> = emptyMap<String, Any>()
        private var lookBefore = ThemeMode.PHONE
        private var galleryBefore = emptySet<String>()
        private var jobsBefore = emptySet<String>()
        private var chatsBefore = emptySet<String>()

        /** A picture this check drew, for the models that change a picture. */
        private var ownPicture: Uri? = null

        private fun prefs() = ctx.getSharedPreferences("anima", Context.MODE_PRIVATE)
        private fun chats() = ChatStore(File(ctx.filesDir, "chats"))

        @BeforeClass @JvmStatic fun start() {
            prefsBefore = prefs().all.toMap()
            lookBefore = Look.mode
            galleryBefore = Gallery.list(ctx).map { it.uri.toString() }.toSet()
            chatsBefore = chats().list().map { it.id }.toSet()
            scenario = ActivityScenario.launch(MainActivity::class.java)
            Queue.init(ctx)
            jobsBefore = Queue.store.all().map { it.id }.toSet()
            assertEquals("the queue must be empty before the check (something is drawing)", 0, Queue.store.waitingCount())
        }

        @AfterClass @JvmStatic fun cleanUp() {
            runCatching { Gallery.deleteMany(ctx, Gallery.list(ctx).filter { it.uri.toString() !in galleryBefore }) }
            runCatching { Queue.store.remove(Queue.store.all().map { it.id }.toSet() - jobsBefore) }
            runCatching { chats().list().map { it.id }.filter { it !in chatsBefore }.forEach { chats().delete(it) } }
            runCatching { Look.set(ctx, lookBefore) }
            val e = prefs().edit().clear()
            for ((k, v) in prefsBefore) when (v) {
                is String -> e.putString(k, v); is Int -> e.putInt(k, v); is Long -> e.putLong(k, v)
                is Boolean -> e.putBoolean(k, v); is Float -> e.putFloat(k, v)
                is Set<*> -> e.putStringSet(k, v.filterIsInstance<String>().toSet())
            }
            e.commit()
            runCatching { scenario.close() }
        }
    }

    private fun log(s: String) = Log.i(TAG, s)

    // ---- finding things on screen ----

    private fun waitFor(sel: BySelector, ms: Long = 10_000): UiObject2? = device.wait(Until.findObject(sel), ms)

    private fun need(sel: BySelector, what: String, ms: Long = 10_000): UiObject2 =
        waitFor(sel, ms) ?: error("not on screen: $what")

    /** Finds something that may be further down a scrolling page: scrolls down, then back up, until it shows. */
    private fun scrollTo(sel: BySelector, what: String): UiObject2 {
        waitFor(sel, 2_000)?.let { return it }
        val w = device.displayWidth / 2
        val h = device.displayHeight
        repeat(8) {
            device.swipe(w, h * 7 / 10, w, h * 3 / 10, 20)
            waitFor(sel, 800)?.let { return it }
        }
        repeat(16) {
            device.swipe(w, h * 3 / 10, w, h * 7 / 10, 20)
            waitFor(sel, 800)?.let { return it }
        }
        error("not on screen, even after scrolling: $what")
    }

    /** Opens a page from the side strip (unfolded) or the side menu (narrow screens). */
    private fun go(page: String) {
        device.findObject(By.desc("Menu"))?.click()
        need(By.res("nav-$page"), "the $page button").click()
        device.waitForIdle()
    }

    /** Picks a model from the Create menu by the name it shows. */
    private fun pickModel(name: String) {
        need(By.res("model-menu"), "the model menu").click()
        val row = device.wait(Until.findObjects(By.textStartsWith(name)), 5_000)?.firstOrNull { !it.text.endsWith("▾") }
            ?: error("\"$name\" is not in the model menu")
        row.click()
        need(By.res("model-menu").textStartsWith(name), "\"$name\" chosen in the model menu")
    }

    private fun setPrompt(text: String) {
        val box = scrollTo(By.res("prompt"), "the prompt box")
        box.text = text
    }

    /** Taps a button that opens the photo picker, handing the app [uri] instead, so the picker never shows. */
    private fun choosePicture(button: BySelector, uri: Uri) {
        Intents.init()
        try {
            intending(not(isInternal())).respondWith(Instrumentation.ActivityResult(Activity.RESULT_OK, Intent().setData(uri)))
            scrollTo(button, "the picture button").click()
            need(By.text("Change"), "the chosen picture")
        } finally { Intents.release() }
    }

    // ---- drawing ----

    private fun newJobs() = Queue.store.all().filter { it.id !in jobsBefore }

    /** Waits for the next job of [model] added after [before] to finish, and checks it really made a picture. */
    private fun waitForJob(model: String, before: Set<String>, minutes: Int): Job {
        val t0 = System.currentTimeMillis()
        val limit = minutes * 60_000L
        var job: Job? = null
        while (System.currentTimeMillis() - t0 < limit) {
            job = Queue.store.all().firstOrNull { it.id !in before && it.model == model }
            if (job != null && (job.state == JobState.DONE || job.state == JobState.FAILED || job.state == JobState.CANCELLED)) break
            Thread.sleep(2_000)
        }
        assertNotNull("$model: no job reached the queue", job)
        assertEquals("$model: ${job!!.error}", JobState.DONE, job.state)
        val uri = Uri.parse(job.resultUri ?: error("$model: done without a picture"))
        checkNotBlank(model, uri)
        log("$model: done in ${job.ms / 1000} s")
        return job
    }

    /** A picture that came out flat (all black, all one colour) counts as a failure; a clip must have some size. */
    private fun checkNotBlank(model: String, uri: Uri) {
        val r = ctx.contentResolver
        val size = r.openFileDescriptor(uri, "r")!!.use { it.statSize }
        assertTrue("$model: the saved file is only $size bytes", size > 20_000)
        val bmp = r.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it) } ?: return   // clips are checked by size
        val px = IntArray(64 * 64)
        android.graphics.Bitmap.createScaledBitmap(bmp, 64, 64, true).getPixels(px, 0, 64, 0, 0, 64, 64)
        val lum = px.map { ((it shr 16 and 255) * 3 + (it shr 8 and 255) * 6 + (it and 255)) / 10.0 }
        val mean = lum.average()
        val spread = Math.sqrt(lum.sumOf { (it - mean) * (it - mean) } / lum.size)
        assertTrue("$model: the picture is flat (spread %.1f)".format(spread), spread > 6)
    }

    /** Taps Draw with the current settings and waits for the picture. */
    private fun draw(model: String, minutes: Int): Job {
        val before = Queue.store.all().map { it.id }.toSet()
        scrollTo(By.text("Draw"), "the Draw button").click()
        val job = waitForJob(model, before, minutes)
        scrollTo(By.textStartsWith("Done in"), "the finished picture on the Create page")
        return job
    }

    private fun installed(id: String) = ModelStore.byId(id).let { it.offered && it.active(ctx).installed(ctx) != null }

    private val menu by lazy { createChoices(File(ModelStore.QWEN.modelDir(ctx), "$NOCTQ_DIR/sp31.bin").isFile) }

    // ---- the checks, in order ----

    @Test fun a_everyPageOpens() {
        go("Create"); need(By.res("model-menu"), "the Create page")
        go("Chat"); waitFor(By.res("message")) ?: need(By.textContains("ownload"), "the Chat page")
        go("Gallery"); need(By.text("Gallery"), "the Gallery page")
        go("Queue"); need(By.text("Queue"), "the Queue page")
        assertTrue("the Queue page did not open", poll(5_000) { device.findObject(By.res("model-menu")) == null } == true)
        device.pressBack()   // Back leaves the queue for the page under it, not the app
        need(By.res("model-menu"), "the Create page after Back on the Queue page")
        go("Settings"); need(By.text("Theme"), "the Settings page")
    }

    @Test fun b_everyModelIsInTheMenu() {
        go("Create")
        need(By.res("model-menu"), "the model menu").click()
        for (c in menu) assertNotNull("\"${c.name}\" is missing from the model menu", waitFor(By.textStartsWith(c.name), 3_000))
        device.pressBack()
        for (c in menu) {
            val st = ModelStore.byId(c.id)
            if (!st.offered) continue
            pickModel(c.name)
            if (!installed(c.id)) need(By.text("Download…"), "the Download button for ${c.name}")
            else if (c.id !in NEEDS_PICTURE) scrollTo(By.text("Draw"), "the Draw button for ${c.name}")
        }
    }

    @Test fun c_everyModelDrawsFromWords() {
        go("Create")
        val prompts = mapOf(
            "anima" to "masterpiece, best quality, safe, 1girl, solo, smile, cherry blossoms, outdoors",
            "klein" to "a cozy coffee shop with plants and warm light, film photo, safe",
            "krea" to "a red fox sitting in fresh snow, morning light, photo, safe",
            "qwen" to "A lighthouse on a rocky cliff at sunset, safe.",
            "moss" to "A small wooden boat on a calm lake at dawn, safe.",
        )
        for (c in menu) {
            if (c.id in NEEDS_PICTURE || !installed(c.id)) continue
            log("drawing with ${c.name}")
            pickModel(c.name)
            setPrompt(prompts[c.id] ?: "a quiet mountain village at dawn, safe")
            val job = draw(c.id, minutes = 10)
            if (ownPicture == null && job.width == job.height) ownPicture = Uri.parse(job.resultUri)
        }
    }

    @Test fun d_picturesCanBeChangedAndAnimated() {
        val pic = ownPicture ?: run { log("no picture drawn on this phone: nothing to change"); return }
        go("Create")
        for (c in menu) {
            if (!installed(c.id)) continue
            val edits = c.id in NEEDS_PICTURE || c.id == "klein" || (c.id == "qwen" && ModelStore.QWEN.installed(ctx) != "small")
            if (!edits) continue
            log("changing a picture with ${c.name}")
            pickModel(c.name)
            setPrompt(if (c.id == "wan") "The light shifts slowly and the leaves move in a breeze." else "Make it night with a starry sky, safe")
            choosePicture(By.text(if (device.hasObject(By.text("Change"))) "Change" else "Choose picture"), pic)
            if (c.id == "wan") {
                val before = Queue.store.all().map { it.id }.toSet()
                scrollTo(By.text("Draw"), "the Draw button").click()
                waitForJob("wan", before, minutes = 15)
            } else draw(c.id, minutes = 10)
            scrollTo(By.text("Remove"), "the Remove button").click()
        }
        if (installed("anima")) {   // Anima follows a guide picture, and its result can be made 4x bigger
            log("Anima with a guide picture, then 4x")
            pickModel("Anima")
            setPrompt("masterpiece, best quality, safe, 1girl, solo, smile, outdoors")
            choosePicture(By.text("Guide picture"), pic)
            draw("anima", minutes = 5)
            scrollTo(By.text("Upscale 4×"), "the Upscale button").click()
            need(By.textStartsWith("Upscaled to"), "the 4x picture", ms = 120_000)
        }
    }

    @Test fun e_chatAnswersAndDraws() {
        if (LocalLing.store(ctx).installed(ctx) == null) { log("no chat model on this phone"); return }
        go("Chat")
        device.findObject(By.desc("New chat"))?.click()
        need(By.res("message"), "the message box").text = "Draw a red fox sitting in the snow, safe"
        need(By.desc("Send"), "the Send button").click()
        need(By.text("Draw this?"), "the picture card in the chat", ms = 300_000)
        val before = Queue.store.all().map { it.id }.toSet()
        need(By.text("Draw"), "the card's Draw button").click()
        val drawn = poll(30_000) { Queue.store.all().firstOrNull { it.id !in before && it.origin == "chat" } }
        val job = waitForJob((drawn ?: error("the card added no job")).model, before, minutes = 10)
        assertEquals("chat", job.origin)
        // and a plain answer, in the same chat
        need(By.res("message"), "the message box").text = "Reply with one word: hello"
        need(By.desc("Send"), "the Send button").click()
        val t0 = System.currentTimeMillis()
        while (System.currentTimeMillis() - t0 < 300_000 && (Chat.busy() || lastAnswer().isNullOrBlank())) Thread.sleep(2_000)
        assertTrue("the chat gave no answer", !lastAnswer().isNullOrBlank())
    }

    private fun lastAnswer(): String? = chats().list().firstOrNull { it.id !in chatsBefore }?.msgs?.lastOrNull()
        ?.let { it as? Msg.Assistant }?.text

    @Test fun f_galleryOpensUsesAndDeletes() {
        val mine = Gallery.list(ctx).filter { it.uri.toString() !in galleryBefore }
        if (mine.isEmpty()) { log("nothing drawn: the gallery check would touch your own pictures"); return }
        go("Gallery")
        need(By.res("gallery-item"), "a picture in the gallery").click()   // the newest: one of the check's own
        need(By.text("Use this prompt"), "the picture's page").click()
        assertEquals("Use this prompt did not fill the prompt", mine.first().prompt, need(By.res("prompt"), "the prompt box").text)
        val count = Gallery.list(ctx).size
        go("Gallery")
        need(By.res("gallery-item"), "a picture in the gallery").click()
        need(By.text("Delete"), "the Delete button").click()
        poll(10_000) { Gallery.list(ctx).size == count - 1 }
        assertEquals("Delete did not remove the picture", count - 1, Gallery.list(ctx).size)
    }

    @Test fun g_themeSwitches() {
        go("Settings")
        for ((label, mode) in listOf("Light" to ThemeMode.LIGHT, "Black" to ThemeMode.BLACK, "Phone" to ThemeMode.PHONE)) {
            need(By.text(label), "the $label theme").click()
            poll(3_000) { Look.mode == mode }
            assertEquals(mode, Look.mode)
        }
    }

    /** Checks [f] every 0.3 s for up to [ms]; returns its first answer that is neither null nor false. */
    private fun <T> poll(ms: Long, f: () -> T?): T? {
        val t0 = System.currentTimeMillis()
        while (System.currentTimeMillis() - t0 < ms) { f()?.let { if (it != false) return it }; Thread.sleep(300) }
        return null
    }
}
