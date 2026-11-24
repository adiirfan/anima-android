package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatToolsTest {
    private val all = setOf("anima", "qwen")
    private val noPicture = "There is no picture to edit."

    @Test fun okDefaults() {
        val r = checkDraw("""{"model":"anima","prompt":"1girl, rain"}""", null, true, all) as ToolCheck.Ok
        assertEquals(DrawArgs("anima", "1girl, rain", "square", false, 1), r.args)
    }
    @Test fun unknownModel() { assertTrue(checkDraw("""{"model":"flux","prompt":"x"}""", null, true, all) is ToolCheck.Bad) }
    @Test fun editNeedsPicture() {
        assertEquals(ToolCheck.Bad(noPicture), checkDraw("""{"model":"qwen","prompt":"x","edit":true}""", noPicture, true, all))
    }
    @Test fun editGoesToQwen() {
        val r = checkDraw("""{"model":"anima","prompt":"x","edit":true}""", null, true, all) as ToolCheck.Ok
        assertEquals("qwen", r.args.model)
    }
    @Test fun smallQwenCannotEdit() { assertTrue(checkDraw("""{"model":"qwen","prompt":"x","edit":true}""", null, false, all) is ToolCheck.Bad) }
    @Test fun notInstalled() { assertTrue(checkDraw("""{"model":"qwen","prompt":"x"}""", null, true, setOf("anima")) is ToolCheck.Bad) }
    @Test fun countClamped() { assertEquals(4, (checkDraw("""{"model":"qwen","prompt":"x","count":9}""", null, true, all) as ToolCheck.Ok).args.count) }
    @Test fun badJson() { assertTrue(checkDraw("""{"model":""", null, true, all) is ToolCheck.Bad) }
}
