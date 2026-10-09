package com.anima.npu

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * The three drawing engines' start-up state, shared by the Create screen and the queue service. Starting an
 * engine opens its files and word tables; the pieces themselves load on the AI chip when a picture needs them, and
 * [release] gives that memory back without forgetting what was started.
 */
object Engines {
    private lateinit var app: Context
    private val lock = Mutex()
    private var anima = false
    private var klein = false
    private var qwen: String? = null   // the Qwen drawing model it was started with ("qwen" or "noctq")
    var memoryMode = 1
        private set
    private var memOverride = -1
    private var loadOpts = -1

    @Synchronized fun init(ctx: Context, memOverride: Int = -1, loadOpts: Int = -1) {
        if (::app.isInitialized && memOverride < 0 && loadOpts < 0) return
        app = ctx.applicationContext
        this.memOverride = memOverride
        this.loadOpts = loadOpts
        memoryMode = app.getSharedPreferences("anima", Context.MODE_PRIVATE).getInt("memoryMode", 1)
    }

    private val modelDir get() = File(app.getExternalFilesDir(null), "models")
    private val libDir get() = app.applicationInfo.nativeLibraryDir

    suspend fun ensureAnima(): String? = lock.withLock {
        withContext(Dispatchers.Default) {
            if (anima) return@withContext null
            Engine.init(modelDir.absolutePath, libDir)?.let { return@withContext it }
            Engine.setMemoryMode(if (memOverride >= 0) memOverride else memoryMode)
            if (loadOpts >= 0) Engine.setLoadOptions(loadOpts)
            Tokens.init(modelDir)
            anima = true
            null
        }
    }

    suspend fun ensureKlein(): String? = lock.withLock {
        withContext(Dispatchers.Default) {
            if (klein) return@withContext null
            val dir = ModelStore.KLEIN.modelDir(app)
            Engine.kleinInit(dir.absolutePath, libDir)?.let { return@withContext it }
            Engine.kleinSetMemoryMode(memoryMode)
            KleinTokens.init(dir)
            klein = true
            null
        }
    }

    /** [drawing]: "qwen" (Qwen-Image 2.1) or "noctq" (qwen/noctq/); switching restarts Qwen with the other files. */
    suspend fun ensureQwen(drawing: String): String? = lock.withLock {
        withContext(Dispatchers.Default) {
            if (qwen == drawing) return@withContext null
            val dir = ModelStore.QWEN.modelDir(app)
            val draw = if (drawing == "noctq") File(dir, NOCTQ_DIR) else dir
            if (qwen != null) Engine.release()   // the other drawing model's pieces leave the AI chip
            qwen = null
            Engine.qwenInit(dir.absolutePath, libDir, draw.absolutePath)?.let { return@withContext it }
            QwenImageTokens.init(dir)
            qwen = drawing
            null
        }
    }

    fun setMemoryMode(m: Int) {
        memoryMode = m
        if (anima) Engine.setMemoryMode(m)
        if (klein) Engine.kleinSetMemoryMode(m)
    }

    /** After a model download: the engine reads its files afresh next time. */
    fun invalidate(model: String) {
        when (model) { "klein" -> klein = false; "qwen" -> qwen = null; else -> anima = false }
    }

    fun release() = Engine.release()
}
