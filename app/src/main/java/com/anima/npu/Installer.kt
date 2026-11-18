package com.anima.npu

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

private fun gb(bytes: Long) = "%.1f GB".format(bytes / 1e9)

/**
 * Runs model downloads (and the check and swap that follow each one) outside any screen, so the user can go to
 * the gallery or the Create screen while they run. Several models download at the same time, each as its own
 * task: the first-run setup starts all the chosen ones together instead of one after another.
 */
object Installer {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val jobs = HashMap<String, Job>()

    /** One model's download, as the screens show it. */
    class Task(val storeId: String) {
        var running by mutableStateOf(true)
            internal set
        var fraction by mutableFloatStateOf(0f)
            internal set
        var status by mutableStateOf("Starting…")
            internal set
        var failed by mutableStateOf(false)
            internal set
    }

    /** Downloads by store id: running ones, and finished or failed ones until the next start of that model. */
    val tasks = mutableStateMapOf<String, Task>()

    /** Whether any download is running. */
    val running get() = tasks.values.any { it.running }

    fun isRunning(storeId: String) = tasks[storeId]?.running == true

    fun task(storeId: String): Task? = tasks[storeId]

    /** (store id, variant id) of installs that just finished; the app picks them up and removes them. */
    val finished = mutableStateListOf<Pair<String, String>>()

    /**
     * [beforeSwap] frees the neural unit's pieces before any file is replaced: called only when the model is
     * already installed (a switch or an update), so a first download never interrupts a picture being drawn.
     */
    fun start(ctx: Context, store: ModelStore, m: ModelStore.Manifest, variant: String, beforeSwap: () -> Unit) {
        if (isRunning(store.id)) return
        val app = ctx.applicationContext
        val t = Task(store.id)
        tasks[store.id] = t
        val job = scope.launch(start = kotlinx.coroutines.CoroutineStart.LAZY) {
            try {
                if (store.installed(app) != null) beforeSwap()
                store.install(app, m, variant) { done, total, what ->
                    t.fraction = if (total > 0) done.toFloat() / total else 1f
                    t.status = "$what… ${gb(done)} of ${gb(total)}"
                }
                t.fraction = 1f
                t.status = "Done."
                finished.add(store.id to variant)
            } catch (e: CancellationException) {
                t.status = "Cancelled."
                t.failed = true
            } catch (e: Exception) {
                t.status = e.message ?: "Download failed"
                t.failed = true
            } finally {
                t.running = false
                synchronized(jobs) { jobs.remove(store.id) }
            }
        }
        synchronized(jobs) { jobs[store.id] = job }
        job.start()
    }

    /** Stops one model's download; its files in Android's download queue are removed as the task ends. */
    fun cancel(storeId: String) {
        synchronized(jobs) { jobs[storeId] }?.cancel()
    }
}
