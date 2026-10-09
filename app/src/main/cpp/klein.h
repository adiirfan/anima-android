#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "qnn_runner.h"

// FLUX.2 [klein] 4B on the neural unit: Qwen3-4B prompt reader in three pieces (text0-2.bin, the hidden states
// after 9 / 18 / 27 layers), the drawing model in four pieces (part0-3.bin, graphs s<W>x<H> for a new picture and
// e<W>x<H> for an edit, sharing weights), taef2 decoder / encoder (dec.bin, enc.bin, one graph per size).
// Word table klein_qwen_q8.raw: 8-bit rows + float row scales, already divided by the reader's residual scale.
struct KleinRequest {
    std::vector<int32_t> ids;      // Qwen ids of the chat-wrapped prompt (no padding), at most 128
    uint64_t seed = 0;
    int steps = 4;
    int width = 512, height = 512;
    std::vector<uint32_t> edit;    // optional width*height ARGB picture to edit; empty = new picture
};

struct KleinResult {
    std::vector<uint32_t> argb;
    double msText = 0, msLoad = 0, msDraw = 0, msDecode = 0;
    std::string error;
};

class KleinPipeline {
public:
    bool init(const std::string& modelDir, const std::string& libDir, std::string* err);
    KleinResult generate(const KleinRequest& req, const std::function<void(int, int)>& progress);
    void release();
    // 0 everything stays loaded, 1 prompt reader and drawing pieces take turns, 2 low (pieces 2-3 swapped in per
    // step), 3 lowest (one drawing piece at a time) - as Anima's modes.
    void setMemoryMode(int m) { mode_ = m; if (m != 0) release(); }
    static std::vector<float> sigmas(int steps, int imageTokens);

private:
    bool runText(const KleinRequest& req, std::vector<float> taps[3], std::string* err);
    bool loadPart(int k, std::string* err);
    int resident() const { return mode_ <= 1 ? 4 : mode_ == 2 ? 2 : 0; }
    int mode_ = 1;
    std::string dir_, graph_;
    int w_ = 0, h_ = 0;
    bool edit_ = false;
    uint64_t spill_ = 0;
    const int8_t* table_ = nullptr;
    const float* scale_ = nullptr;
    size_t tableBytes_ = 0;
    QnnGraph text_[3], part_[4], dec_;
};
