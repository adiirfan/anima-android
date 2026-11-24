package com.anima.npu

import org.junit.Assert.assertEquals
import org.junit.Test

class CoresTest {
    private fun cores(vararg groups: Pair<Int, Long>) = groups.flatMap { (n, f) -> List(n) { f } }

    @Test fun fourFastFourSlow() = assertEquals(4, fastCores(cores(4 to 1_766_400L, 4 to 2_803_200L)))   // XZ3

    @Test fun oneTopThreeBigFourLittle() = assertEquals(4, fastCores(cores(4 to 1_800_000L, 3 to 2_400_000L, 1 to 3_000_000L)))

    @Test fun twoBigSixLittle() = assertEquals(2, fastCores(cores(6 to 1_800_000L, 2 to 2_200_000L)))

    @Test fun allTheSame() = assertEquals(8, fastCores(cores(8 to 2_000_000L)))

    @Test fun unreadable() = assertEquals(4, fastCores(emptyList()))

    @Test fun unreadableCoresIgnored() = assertEquals(4, fastCores(cores(4 to 0L, 4 to 2_803_200L, 4 to 1_766_400L)))
}
