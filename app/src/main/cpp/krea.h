#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "qnn_runner.h"

// Krea 2 Turbo (12B) on the neural unit (build: phone-v81/krea/build/krea). Qwen3-VL-4B prompt reader in three pieces
// (text0-2.bin, 8-bit, 128 slots, each giving 4 of Krea's 12 tapped layers), the prompt fusion (fuse.bin), the drawing
// model one block per file (b0-27.bin, graph s<W>x<H>_<i>), the taew2_1 decoder (dec.bin, graph s<W>x<H>). Files:
// krea_qwen_q8.raw (word table: 8-bit rows + float row scales, already divided by the reader's residual scale),
// mods.f32 (the modulations of Turbo's 8 fixed noise levels, (8, 29, 6, 6144)), sigmas.txt.
struct KreaRequest {
    std::vector<int32_t> ids;      // Qwen ids of the whole Krea template (system turn, prompt, assistant opening)
    uint64_t seed = 0;
    int width = 768, height = 768;
};

struct KreaResult {
    std::vector<uint32_t> argb;
    double msText = 0, msDraw = 0, msDecode = 0, msWait = 0;
    std::string error;
};

class KreaPipeline {
public:
    bool init(const std::string& modelDir, const std::string& libDir, std::string* err);
    KreaResult generate(const KreaRequest& req, const std::function<void(int, int)>& progress);
    void release();

private:
    bool runText(const KreaRequest& req, std::vector<float>* txt, std::vector<float>* kbText, std::string* err);
    std::string dir_;
    std::vector<float> sig_;
    const int8_t* table_ = nullptr;
    const float* scale_ = nullptr;
    size_t tableBytes_ = 0;
    const float* mods_ = nullptr;
    size_t modsBytes_ = 0;
    QnnGraph dec_;
};
