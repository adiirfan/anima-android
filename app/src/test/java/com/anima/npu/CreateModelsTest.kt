package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Every picture model the app lets people download must be usable on the Create page (Krea was not, 2026-11-24). */
class CreateModelsTest {
    private val menuIds = createChoices(noctq = true).map { it.id }.toSet()
    private val downloadable = (ModelStore.SETUP_PICTURE + ModelStore.PICTURE).map { it.id }.toSet()

    @Test fun everyDownloadableModelIsInTheMenu() =
        assertEquals("downloadable but missing from the Create menu", emptySet<String>(), downloadable - menuIds)

    @Test fun everyMenuModelCanBeDownloaded() =
        assertEquals("in the Create menu but not downloadable", emptySet<String>(), menuIds - downloadable)

    @Test fun menuNamesAreDistinct() {
        val names = createChoices(noctq = true).map { it.name }
        assertEquals(names.size, names.toSet().size)
    }

    @Test fun noctQOnlyWhenPresent() {
        assertTrue(createChoices(noctq = false).none { it.drawing == "noctq" })
        assertTrue(createChoices(noctq = true).any { it.drawing == "noctq" })
    }

    @Test fun pictureGroupsAreMenuModels() {
        assertEquals(emptySet<String>(), (NEEDS_PICTURE + TEXT_ONLY) - menuIds)
        assertEquals(emptySet<String>(), NEEDS_PICTURE intersect TEXT_ONLY)
    }

    /** A model with no prompt slot of its own silently shares (and overwrites) Anima's. */
    @Test fun everyModelKeepsItsOwnPrompt() {
        val s = CreateState()
        for (id in menuIds) { s.model = id; s.prompt = "prompt for $id" }
        for (id in menuIds) { s.model = id; assertEquals("prompt for $id", s.prompt) }
    }

    /** A model with no name of its own shows as "Anima" on the Queue page. */
    @Test fun everyModelHasItsOwnQueueName() {
        for (id in menuIds - "anima") {
            val j = Job("j", 1, id, "qwen", "p", 1, 8, 0, 0, 1f, 0.5f, null, "create")
            assertNotEquals("$id shows as Anima in the queue", "Anima", modelName(j))
        }
    }
}
