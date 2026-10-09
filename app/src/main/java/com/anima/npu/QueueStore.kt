package com.anima.npu

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import java.io.File

/**
 * The picture queue on disk (dir/queue.json), rewritten whole and atomically on every change, so it survives the
 * app being closed or killed. [changed] ticks on every change, for screens that show the queue.
 */
class QueueStore(private val dir: File) {
    private val file = File(dir, "queue.json")
    private var jobs: MutableList<Job> = load()
    private val _changed = MutableStateFlow(0)
    val changed: StateFlow<Int> = _changed

    private fun load(): MutableList<Job> {
        if (!file.isFile) return mutableListOf()
        val a = JSONArray(file.readText())
        return MutableList(a.length()) { Job.fromJson(a.getJSONObject(it)) }
    }

    private fun save() {
        dir.mkdirs()
        val tmp = File(dir, "queue.json.tmp")
        tmp.writeText(JSONArray().apply { jobs.forEach { put(it.toJson()) } }.toString())
        tmp.renameTo(file)
        _changed.value = _changed.value + 1
    }

    @Synchronized fun all(): List<Job> = jobs.toList()
    @Synchronized fun get(id: String): Job? = jobs.firstOrNull { it.id == id }
    @Synchronized fun add(j: Job) { jobs.add(j); save() }
    @Synchronized fun update(id: String, f: (Job) -> Job) {
        val i = jobs.indexOfFirst { it.id == id }
        if (i < 0) return
        jobs[i] = f(jobs[i]); save()
    }
    @Synchronized fun remove(ids: Set<String>) { if (jobs.removeAll { it.id in ids }) save() }
    @Synchronized fun nextWaiting(): Job? = jobs.firstOrNull { it.state == JobState.WAITING }
    /** Jobs still to draw, including the one drawing now. */
    @Synchronized fun waitingCount(): Int = jobs.count { it.state == JobState.WAITING || it.state == JobState.DRAWING }

    /** A job left "drawing" by a killed app is drawn again. */
    @Synchronized fun recoverInterrupted() {
        if (jobs.none { it.state == JobState.DRAWING }) return
        // a second interruption is almost always Android closing the app for memory: stop rather than loop
        jobs = jobs.map {
            when {
                it.state != JobState.DRAWING -> it
                it.tries >= 1 -> it.copy(state = JobState.FAILED, error = "The phone ran out of memory while drawing this " +
                    "in the background. Keep Anima open on screen for this one (picture edits need about 5 GB), then Retry.")
                else -> it.copy(state = JobState.WAITING, tries = it.tries + 1)
            }
        }.toMutableList()
        save()
    }

    /** Finished, failed and cancelled jobs older than [keepMs] leave the list (their pictures stay in the gallery). */
    @Synchronized fun prune(now: Long, keepMs: Long = 7 * 86_400_000L) {
        val before = jobs.size
        jobs = jobs.filter { it.state == JobState.WAITING || it.state == JobState.DRAWING || now - it.created < keepMs }.toMutableList()
        if (jobs.size != before) save()
    }
}

/** The app's one queue, opened by whichever starts first: the screen or the queue service. */
object Queue {
    lateinit var store: QueueStore
        private set
    @Synchronized fun init(ctx: Context) {
        if (::store.isInitialized) return
        store = QueueStore(dir(ctx)).apply { recoverInterrupted(); prune(System.currentTimeMillis()) }
    }
    fun dir(ctx: Context) = File(ctx.filesDir, "queue").apply { mkdirs() }
}
