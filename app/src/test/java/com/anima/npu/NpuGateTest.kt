package com.anima.npu

import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Collections

class NpuGateTest {
    @Test fun neverTwoAtOnceAndAWaitingChatGoesFirst() = runBlocking {
        val g = Gate()
        val log = Collections.synchronizedList(mutableListOf<String>())
        val a = async { g.withQueue { log += "q1+"; delay(100); log += "q1-" } }
        delay(10)
        val b = async { g.withQueue { log += "q2+"; delay(10); log += "q2-" } }
        delay(10)
        val c = async { g.withChat { log += "c+"; delay(10); log += "c-" } }
        a.await(); b.await(); c.await()
        assertEquals(listOf("q1+", "q1-", "c+", "c-", "q2+", "q2-"), log)
    }

    @Test fun holderShowsWhoHasIt() = runBlocking {
        val g = Gate()
        assertEquals("idle", g.holder.value)
        g.withChat { assertEquals("chat", g.holder.value) }
        g.withQueue { assertEquals("queue", g.holder.value) }
        assertEquals("idle", g.holder.value)
    }
}
