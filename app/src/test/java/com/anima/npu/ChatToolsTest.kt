package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatToolsTest {
    private val all = setOf("anima", "qwen")
    @Test fun okDefaults() {
        val r = checkDraw("""{"model":"anima","prompt":"1girl, rain"}""", false, true, all) as ToolCheck.Ok
        assertEquals(DrawArgs("anima", "1girl, rain", "square", false, 1), r.args)
    }
    @Test fun unknownModel() { assertTrue(checkDraw("""{"model":"flux","prompt":"x"}""", false, true, all) is ToolCheck.Bad) }
    @Test fun editNeedsPicture() { assertTrue(checkDraw("""{"model":"qwen","prompt":"x","edit":true}""", false, true, all) is ToolCheck.Bad) }
    @Test fun editOnlyQwen() { assertTrue(checkDraw("""{"model":"anima","prompt":"x","edit":true}""", true, true, all) is ToolCheck.Bad) }
    @Test fun smallQwenCannotEdit() { assertTrue(checkDraw("""{"model":"qwen","prompt":"x","edit":true}""", true, false, all) is ToolCheck.Bad) }
    @Test fun notInstalled() { assertTrue(checkDraw("""{"model":"qwen","prompt":"x"}""", false, true, setOf("anima")) is ToolCheck.Bad) }
    @Test fun countClamped() { assertEquals(4, (checkDraw("""{"model":"qwen","prompt":"x","count":9}""", false, true, all) as ToolCheck.Ok).args.count) }
    @Test fun badJson() { assertTrue(checkDraw("""{"model":""", false, true, all) is ToolCheck.Bad) }
}
