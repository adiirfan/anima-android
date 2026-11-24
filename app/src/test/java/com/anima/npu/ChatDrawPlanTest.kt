package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatDrawPlanTest {
    private val both = setOf("anima", "qwen")

    // what the user named in their own words
    @Test fun nothingNamed() = assertEquals(Named(null, null, null), namedSettings("draw a safe picture of a cat in a garden"))
    @Test fun modelNamed() = assertEquals("qwen", namedSettings("a photo of a cat, use Qwen").model)
    @Test fun noctQIsQwen() = assertEquals("qwen", namedSettings("draw it with noct q").model)
    @Test fun animaNamed() = assertEquals("anima", namedSettings("anima: a cat").model)
    @Test fun largeSize() = assertEquals(1, namedSettings("draw a large picture of a cat").tier)
    @Test fun sizeNumber() = assertEquals(1, namedSettings("a cat at 768 please").tier)
    @Test fun highRes() = assertEquals(1, namedSettings("High-res cat in a garden").tier)
    @Test fun smallSize() = assertEquals(0, namedSettings("a cat, 512").tier)
    @Test fun bigInThePictureIsNotSize() = assertEquals(null, namedSettings("draw a big dog next to a large tree").tier)
    @Test fun stepsNamed() = assertEquals(12, namedSettings("draw a cat, 12 steps").steps)
    @Test fun stepsCapped() = assertEquals(30, namedSettings("use 99 steps").steps)
    @Test fun zeroStepsIgnored() = assertEquals(null, namedSettings("0 steps").steps)

    // asking for a picture, in the user's words
    @Test fun drawIsAsking() = assertTrue(wantsPicture("draw an anime cat in a garden, safe"))
    @Test fun makeAPicture() = assertTrue(wantsPicture("can you make me a picture of a beach at sunset"))
    @Test fun pictureOf() = assertTrue(wantsPicture("I want an image of a red car"))
    @Test fun greetingIsNot() = assertFalse(wantsPicture("hello, how are you?"))
    @Test fun questionIsNot() = assertFalse(wantsPicture("how do I make a picture bigger?"))
    @Test fun whatCanYouDrawIsNot() = assertFalse(wantsPicture("what can you draw?"))

    @Test fun wordsWithoutTheAsk() = assertEquals("anime cat in a garden, safe", pictureWords("draw an anime cat in a garden, safe"))
    @Test fun wordsWithoutPictureOf() = assertEquals("beach at sunset", pictureWords("Can you make me a picture of a beach at sunset?"))
    @Test fun wordsNothingLeft() = assertEquals("", pictureWords("draw something"))

    // what happens: draw now, ask with a card, or refuse
    @Test fun nothingNamedAsksAll() = assertEquals(DrawPlan.Ask("anima", askModel = true, askSize = true, askSteps = true, tier = 0, steps = 4),
        planDraw("anima", edit = false, Named(null, null, null), both))

    @Test fun suggestedModelMissingFallsBack() = assertEquals("anima",
        (planDraw("qwen", false, Named(null, null, null), setOf("anima")) as DrawPlan.Ask).model)

    @Test fun oneModelNoModelRow() = assertFalse((planDraw("anima", false, Named(null, null, null), setOf("anima")) as DrawPlan.Ask).askModel)

    @Test fun allNamedDrawsNow() = assertEquals(DrawPlan.Queue("anima", 1, 8), planDraw("qwen", false, Named("anima", 1, 8), both))

    @Test fun qwenNamedDrawsNow() = assertEquals(DrawPlan.Queue("qwen", 0, 8), planDraw("anima", false, Named("qwen", null, null), both))

    @Test fun onlyQwenDownloadedDrawsNow() = assertEquals(DrawPlan.Queue("qwen", 0, 8), planDraw("anima", false, Named(null, null, null), setOf("qwen")))

    @Test fun partlyNamedAsksTheRest() = assertEquals(DrawPlan.Ask("anima", askModel = false, askSize = false, askSteps = true, tier = 1, steps = 4),
        planDraw("qwen", false, Named("anima", 1, null), both))

    @Test fun editsDrawNow() = assertEquals(DrawPlan.Queue("qwen", 0, 8), planDraw("qwen", true, Named(null, null, null), both))

    @Test fun namedModelMissingRefused() = assertTrue(planDraw("anima", false, Named("qwen", null, null), setOf("anima")) is DrawPlan.Bad)

    @Test fun nothingDownloadedRefused() = assertTrue(planDraw("anima", false, Named(null, null, null), emptySet()) is DrawPlan.Bad)

    @Test fun otherModelsIgnored() = assertFalse((planDraw("anima", false, Named(null, null, null), setOf("anima", "klein", "moss")) as DrawPlan.Ask).askModel)
}

class SettleOfferTest {
    private val choice = Msg.Choice("c", "cat", "square", 1, "anima", true, true, true, 0, 4)
    private var n = 0
    private fun id() = "n${n++}"

    @Test fun replyAfterCardBecomesThePointer() = assertEquals(
        listOf(Msg.User("u", "draw a cat", null), Msg.ToolCall("t", "k", "draw_picture", "{}"), choice,
            Msg.ToolResult("r", "k", "{}"), Msg.Assistant("n0", OFFER_LINE, "")),
        settleOffer(listOf(Msg.User("u", "draw a cat", null), Msg.ToolCall("t", "k", "draw_picture", "{}"), choice,
            Msg.ToolResult("r", "k", "{}"), Msg.Assistant("a", "The picture is on its way!", "")), ::id))

    @Test fun replyBeforeCardIsKept() = assertEquals(
        listOf(Msg.User("u", "draw a cat", null), Msg.Assistant("a", "Sure!", ""), choice, Msg.Assistant("n0", OFFER_LINE, "")),
        settleOffer(listOf(Msg.User("u", "draw a cat", null), Msg.Assistant("a", "Sure!", ""), choice), ::id))

    @Test fun noCardNoChange() {
        val m = listOf(Msg.User("u", "hi", null), Msg.Assistant("a", "Hello!", ""))
        assertEquals(m, settleOffer(m, ::id))
    }
}

class DrawnChoiceTest {
    private val choice = Msg.Choice("c", "cat", "square", 1, "anima", true, true, true, 0, 4)
    private val pic = Msg.Picture("p", "job1")

    @Test fun cardBecomesPictureAndPointerGoes() = assertEquals(
        listOf(Msg.User("u", "draw a cat", null), pic),
        drawnChoice(listOf(Msg.User("u", "draw a cat", null), choice, Msg.Assistant("a", OFFER_LINE, "")), "c", listOf(pic)))

    @Test fun otherRepliesStay() = assertEquals(
        listOf(Msg.Assistant("a", "Sure!", ""), pic),
        drawnChoice(listOf(Msg.Assistant("a", "Sure!", ""), choice), "c", listOf(pic)))
}

class DrawnChoiceOtherCardTest {
    @Test fun otherCardKeepsItsLine() {
        val a = Msg.Choice("c1", "cat", "square", 1, "anima", true, true, true, 0, 4)
        val b = Msg.Choice("c2", "dog", "square", 1, "anima", true, true, true, 0, 4)
        val pic = Msg.Picture("p", "job1")
        assertEquals(
            listOf(Msg.User("u1", "draw a cat", null), pic, Msg.User("u2", "draw a dog", null), b, Msg.Assistant("l2", OFFER_LINE, "")),
            drawnChoice(listOf(Msg.User("u1", "draw a cat", null), a, Msg.Assistant("l1", OFFER_LINE, ""),
                Msg.User("u2", "draw a dog", null), b, Msg.Assistant("l2", OFFER_LINE, "")), "c1", listOf(pic)))
    }
}
