package com.anima.npu

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * Works through the picture queue one job at a time in the background, with a progress notification while it
 * draws and a "Picture ready" notification (with a preview) for each finished picture. Started whenever a job is
 * added; stops itself when the queue is empty.
 */
class QueueService : Service() {
    companion object {
        @Volatile var running = false
            private set
        /** (job id, text) of the job drawing now, e.g. "step 3 of 4"; empty when idle. */
        private val _progress = MutableStateFlow("" to "")
        val progress: StateFlow<Pair<String, String>> = _progress
        private const val CH_WORK = "queue"
        private const val CH_DONE = "done"
        private const val ID_WORK = 1
        private const val ACTION_CANCEL = "cancel"
        /** Starts the worker. Android only allows this while the app is on screen; otherwise the jobs wait for the
         *  next time it is opened (MainActivity kicks again on resume). */
        fun kick(ctx: Context) {
            runCatching { ctx.startForegroundService(Intent(ctx, QueueService::class.java)) }
                .onFailure { Log.w("queue", "could not start the queue worker now: ${it.message}") }
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var loop: kotlinx.coroutines.Job? = null
    private lateinit var nm: NotificationManager

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        Queue.init(this)
        Engines.init(this)
        nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CH_WORK, "Drawing", NotificationManager.IMPORTANCE_LOW))
        nm.createNotificationChannel(NotificationChannel(CH_DONE, "Finished pictures", NotificationManager.IMPORTANCE_DEFAULT))
    }

    private fun openApp(uri: String? = null): PendingIntent = PendingIntent.getActivity(this, uri?.hashCode() ?: 0,
        Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP).apply { uri?.let { putExtra("openUri", it) } },
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)

    private fun working(title: String, step: Int, total: Int): Notification =
        Notification.Builder(this, CH_WORK).setSmallIcon(android.R.drawable.ic_menu_gallery)
            .setContentTitle(title).setOngoing(true).setOnlyAlertOnce(true)
            .setProgress(total, step, total == 0).setContentIntent(openApp())
            .addAction(Notification.Action.Builder(null, "Cancel the rest",
                PendingIntent.getService(this, 1, Intent(this, QueueService::class.java).setAction(ACTION_CANCEL),
                    PendingIntent.FLAG_IMMUTABLE)).build())
            .build()

    private fun finished(j: Job, bmp: Bitmap, uri: String) {
        val n = Notification.Builder(this, CH_DONE).setSmallIcon(android.R.drawable.ic_menu_gallery)
            .setContentTitle("Picture ready").setContentText(j.prompt.take(80)).setLargeIcon(bmp)
            .setStyle(Notification.BigPictureStyle().bigPicture(bmp)).setAutoCancel(true).setGroup("done")
            .setContentIntent(openApp(uri)).build()
        nm.notify(j.id.hashCode(), n)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(ID_WORK, working("Getting ready…", 0, 0), ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        if (intent?.action == ACTION_CANCEL) {
            Queue.store.all().filter { it.state == JobState.WAITING }
                .forEach { w -> Queue.store.update(w.id) { it.copy(state = JobState.CANCELLED) } }
        }
        if (loop?.isActive != true) loop = scope.launch { work() }
        return START_NOT_STICKY
    }

    private suspend fun work() {
        running = true
        val wl = (getSystemService(POWER_SERVICE) as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "anima:queue")
            .apply { setReferenceCounted(false) }   // taken both while drawing and while Ling answers
        var done = 0
        try {
            while (true) {
                // while Ling answers, stay up (and keep the processor awake) so the reply is not cut off when the
                // screen locks, and so pictures it asks for start at once
                val j = Queue.store.nextWaiting() ?: if (Chat.busy()) {
                    _progress.value = "" to ""
                    nm.notify(ID_WORK, working("${LocalLing.name(this)} is answering…", 0, 0))
                    if (!wl.isHeld) wl.acquire(10 * 60_000L)
                    delay(300)
                    continue
                } else break
                // recounted on every update, so jobs added while this one draws show up in "k of n"
                fun title() = "Drawing ${done + 1} of ${done + Queue.store.waitingCount()}"
                Queue.store.update(j.id) { it.copy(state = JobState.DRAWING) }
                val first = if (NpuGate.gate.holder.value == "chat") "waiting for the chat" else "starting"
                _progress.value = j.id to first
                nm.notify(ID_WORK, working("${title()} · $first", 0, 0))
                wl.acquire(30 * 60_000L)
                try {
                    NpuGate.gate.withQueue {
                        val step: (Int, Int) -> Unit = { i, n ->
                            val what = if (i == 0) "reading the prompt" else "step $i of $n"
                            _progress.value = j.id to what
                            nm.notify(ID_WORK, working("${title()} · $what", i, n))
                        }
                        // Ling's server holds ~4.5 GB. On a 12 GB phone a picture beside it gets the whole app killed
                        // by Android's memory manager (seen on the S25 Edge loading Anima), so close Ling first; the
                        // chat starts it again (about 10 s) with the next message.
                        if (LocalLing.running() && smallMemory()) { Log.i("queue", "closing Ling to draw"); LocalLing.stop() }
                        val r = try { Drawer.draw(this, j, step) } catch (e: Exception) {
                            // Ling's server holds ~4.5 GB: on a 12 GB phone the picture may not fit beside it. Stop
                            // Ling (the chat starts it again on the next message) and try once more.
                            if (!LocalLing.running()) throw e
                            Log.i("queue", "draw failed beside Ling (${e.message}); stopping Ling and retrying")
                            LocalLing.stop()
                            Engine.release()
                            Drawer.draw(this, j, step)
                        }
                        Queue.store.update(j.id) {
                            it.copy(state = JobState.DONE, resultUri = r.entry.uri.toString(), ms = r.ms,
                                width = r.bitmap.width, height = r.bitmap.height)
                        }
                        finished(j, r.bitmap, r.entry.uri.toString())
                    }
                } catch (e: Exception) {
                    Queue.store.update(j.id) { it.copy(state = JobState.FAILED, error = e.message ?: "failed") }
                } finally {
                    if (wl.isHeld) wl.release()
                    done++
                }
            }
        } finally {
            if (wl.isHeld) wl.release()
            // the queue is empty: on a 12 GB phone give the picture models' memory back (the next picture reloads
            // them, a few seconds), or a chat message right after starts Ling on top of them
            if (smallMemory() && Queue.store.waitingCount() == 0) Engines.release()
            running = false
            _progress.value = "" to ""
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun smallMemory(): Boolean = smallMemory(this)

    override fun onDestroy() { scope.cancel(); super.onDestroy() }
}
