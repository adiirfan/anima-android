package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class QueueStoreTest {
    @get:Rule val tmp = TemporaryFolder()
    private fun job(id: String, created: Long = 1) = Job(id, created, "qwen", "noctq", "a cat", 42, 4, 0, 0, 0.8f, 0.5f, null, "create")

    @Test fun addAndReloadKeepsOrderAndFields() {
        val s = QueueStore(tmp.root); s.add(job("a")); s.add(job("b"))
        val t = QueueStore(tmp.root).all()
        assertEquals(listOf("a", "b"), t.map { it.id }); assertEquals("noctq", t[0].qwenDrawing); assertEquals(0.8f, t[0].guideStrength)
    }
    @Test fun nextWaitingSkipsOthers() {
        val s = QueueStore(tmp.root); s.add(job("a")); s.add(job("b"))
        s.update("a") { it.copy(state = JobState.DONE) }
        assertEquals("b", s.nextWaiting()!!.id)
    }
    @Test fun interruptedDrawingGoesBackToWaiting() {
        val s = QueueStore(tmp.root); s.add(job("a")); s.update("a") { it.copy(state = JobState.DRAWING) }
        QueueStore(tmp.root).apply { recoverInterrupted(); assertEquals(JobState.WAITING, all()[0].state) }
        // killed a second time while drawing (Android closing the app for memory): it fails instead of looping
        QueueStore(tmp.root).apply {
            update(all()[0].id) { it.copy(state = JobState.DRAWING) }
            recoverInterrupted()
            assertEquals(JobState.FAILED, all()[0].state)
        }
    }
    @Test fun pruneDropsOldFinishedOnly() {
        val s = QueueStore(tmp.root); s.add(job("old", 0)); s.add(job("wait", 0)); s.add(job("new", 9_000_000_000))
        s.update("old") { it.copy(state = JobState.DONE) }; s.update("new") { it.copy(state = JobState.DONE) }
        s.prune(now = 9_000_000_001, keepMs = 1000)
        assertEquals(listOf("wait", "new"), s.all().map { it.id })
    }
    @Test fun nullableFieldsRoundTrip() {
        val s = QueueStore(tmp.root); s.add(job("a").copy(sourceFile = "a-src.png")); s.update("a") { it.copy(state = JobState.FAILED, error = "boom") }
        val j = QueueStore(tmp.root).all()[0]
        assertEquals("a-src.png", j.sourceFile); assertEquals("boom", j.error); assertNull(j.resultUri)
    }
    @Test fun emptyWhenNoFile() { assertNull(QueueStore(tmp.root).nextWaiting()) }
}
