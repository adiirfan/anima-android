package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ChatStoreTest {
    @get:Rule val tmp = TemporaryFolder()
    @Test fun roundTripsEveryMessageKind() {
        val s = ChatStore(tmp.root)
        val c = Conversation("c1", "Cats", 5, true, listOf(
            Msg.User("1", "draw a cat", "c1-1.png"), Msg.Assistant("2", "Sure", "they want a cat"),
            Msg.ToolCall("3", "call_0", "draw_picture", """{"model":"anima"}"""), Msg.ToolResult("4", "call_0", """{"queued":1}"""),
            Msg.Picture("5", "job1")))
        s.save(c)
        assertEquals(c, ChatStore(tmp.root).get("c1"))
    }
    @Test fun listNewestFirstAndDelete() {
        val s = ChatStore(tmp.root)
        s.save(Conversation("a", "A", 1, false, emptyList())); s.save(Conversation("b", "B", 2, false, emptyList()))
        assertEquals(listOf("b", "a"), s.list().map { it.id })
        s.delete("b"); assertEquals(listOf("a"), s.list().map { it.id })
    }
    @Test fun rename() {
        val s = ChatStore(tmp.root)
        s.save(Conversation("a", "A", 1, false, emptyList())); s.rename("a", "Dogs")
        assertEquals("Dogs", s.get("a")!!.title)
    }
}
