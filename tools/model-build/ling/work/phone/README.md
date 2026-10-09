# Ling-3.0-tiny for the phone's AI chip

`Ling-3.0-tiny-Q4_0-npu2.gguf` is [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
converted for llama.cpp's Hexagon backend: 4-bit (Q4_0), with the 16-row and per-head attention weights kept in f16
so every matrix runs on the chip. Used by the Anima app's chat. Recipe: `tools/model-build/ling/make_ling.sh` in the
app's source.

Ling-3.0-tiny is released by inclusionAI under the MIT licence; this conversion keeps that licence.
