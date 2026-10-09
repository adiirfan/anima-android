package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SseParserTest {
    @Test fun parsesTextThinkingToolAndDone() {
        val p = SseAccumulator()
        assertEquals(Event.Thinking("hmm"), p.feed("""data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}"""))
        assertEquals(Event.Text("Hi"), p.feed("""data: {"choices":[{"delta":{"content":"Hi"}}]}"""))
        assertEquals(Event.Tool(0, "call_1", "draw_picture", "{\"model\""),
            p.feed("""data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"draw_picture","arguments":"{\"model\""}}]}}]}"""))
        assertEquals(Event.Tool(0, null, null, ":\"qwen\"}"),
            p.feed("""data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"qwen\"}"}}]}}]}"""))
        assertNull(p.feed(""))
        assertNull(p.feed(": keep-alive"))
        assertNull(p.feed("""data: {"choices":[{"delta":{"content":null}}]}"""))
        assertEquals(Event.Done, p.feed("data: [DONE]"))
    }
}
