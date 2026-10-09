package com.anima.npu

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
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
 * Runs a model download (and the check and swap that follow it) outside any screen, so the user can go to
 * the gallery or the Create screen while it runs. Android's DownloadManager fetches the files either way;
 * this is what checks them and puts them in place, which used to stop when the download screen was left.
 */
object Installer {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var job: Job? = null

    var running by mutableStateOf(false)
        private set
    var fraction by mutableFloatStateOf(0f)
        private set
    var status by mutableStateOf("")
        private set

    /** The model (store id) being installed, while [running]. */
    var storeId by mutableStateOf<String?>(null)
        private set

    /** (store id, variant id) of an install that just finished; the app picks it up and sets it back to null. */
    var finished by mutableStateOf<Pair<String, String>?>(null)

    /** [beforeSwap] frees the neural unit's pieces before any file is replaced. */
    fun start(ctx: Context, store: ModelStore, m: ModelStore.Manifest, variant: String, beforeSwap: () -> Unit) {
        if (running) return
        val app = ctx.applicationContext
        running = true
        storeId = store.id
        fraction = 0f
        status = "Starting…"
        job = scope.launch {
            try {
                beforeSwap()
                store.install(app, m, variant) { done, total, what ->
                    fraction = if (total > 0) done.toFloat() / total else 1f
                    status = "$what… ${gb(done)} of ${gb(total)}"
                }
                status = "Done."
                finished = store.id to variant
            } catch (e: CancellationException) {
                status = "Cancelled."
            } catch (e: Exception) {
                status = e.message ?: "Download failed"
            } finally {
                running = false
            }
        }
    }

    fun cancel(ctx: Context) {
        ModelStore.ANIMA.cancelAll(ctx.applicationContext)
        job?.cancel()
    }
}
