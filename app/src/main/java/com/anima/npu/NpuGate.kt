package com.anima.npu

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex

/**
 * The AI chip goes to the picture queue or to Ling, never both at once (each would need its own working memory on
 * it). A chat that is waiting goes before the queue's next picture; a picture already drawing is never interrupted.
 */
class Gate {
    private val m = Mutex()
    @Volatile private var chatWaiting = 0
    private val _holder = MutableStateFlow("idle")
    /** "idle", "queue" or "chat". */
    val holder: StateFlow<String> = _holder
    fun chatWaiting() = chatWaiting > 0

    suspend fun <T> withQueue(block: suspend () -> T): T {
        while (true) {
            m.lock()
            // a chat that is waiting goes first: step aside and try again after it
            if (chatWaiting > 0) { m.unlock(); delay(50); continue }
            break
        }
        _holder.value = "queue"
        try { return block() } finally { _holder.value = "idle"; m.unlock() }
    }

    suspend fun <T> withChat(block: suspend () -> T): T {
        synchronized(this) { chatWaiting++ }
        try { m.lock() } finally { synchronized(this) { chatWaiting-- } }
        _holder.value = "chat"
        try { return block() } finally { _holder.value = "idle"; m.unlock() }
    }
}

object NpuGate { val gate = Gate() }
