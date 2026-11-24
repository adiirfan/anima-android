package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Test

class QueueTextTest {
    private fun job(state: JobState, error: String? = null) =
        Job("j1", 1, "anima", "qwen", "safe, 1girl", 42, 8, 0, 0, 1f, 0.5f, null, "create", state = state, error = error)

    @Test fun failedShowsReason() =
        assertEquals("Failed · Out of memory", stateText(job(JobState.FAILED, error = "Out of memory"), "" to ""))

    @Test fun failedWithoutReason() = assertEquals("Failed", stateText(job(JobState.FAILED, error = null), "" to ""))

    @Test fun failedBlankReason() = assertEquals("Failed", stateText(job(JobState.FAILED, error = "  "), "" to ""))

    @Test fun waitingUnchanged() = assertEquals("Waiting", stateText(job(JobState.WAITING), "" to ""))
}

class QueuePagingTest {
    @Test fun fewJobsNoButton() = assertEquals(null, moreLabel(total = 12, shown = 30))
    @Test fun manyJobsNextPage() = assertEquals("Show 30 more", moreLabel(total = 200, shown = 30))
    @Test fun lastPageIsSmaller() = assertEquals("Show 5 more", moreLabel(total = 65, shown = 60))
    @Test fun drawingShowsProgressAlone() = assertEquals("Reading the prompt", drawingLine("reading the prompt"))
    @Test fun drawingWithoutProgress() = assertEquals("Drawing…", drawingLine(""))
}
