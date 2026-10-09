#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "qnn_runner.h"

struct GenRequest {
    std::vector<int32_t> qwenIds;  // Qwen tokenizer ids, no start/end tokens
    std::vector<int32_t> t5Ids;    // T5 tokenizer ids including the end token 1
    std::vector<float> t5Weights;  // emphasis per T5 id ("(word:1.2)"); empty = all 1
    uint64_t seed = 0;
    int steps = 8;
    int width = 512, height = 512;  // one of Pipeline::sizes()
    std::vector<uint32_t> guide;   // optional width*height ARGB guide picture, already cropped and scaled; empty = none
    float guideStrength = 1.f;
    float guideEnd = 0.5f;         // the guide steers the first part of the steps (ComfyUI's end_percent)
    std::vector<float> noise;      // optional 16*(h/8)*(w/8) starting noise (self-test); empty = from seed
};

struct GenResult {
    std::vector<uint32_t> argb;    // width*height ARGB_8888
    std::vector<float> latent;     // final 16*(h/8)*(w/8) latent
    double msLoad = 0, msText = 0, msDraw = 0, msDecode = 0;
    std::string error;
};

// Runs a whole picture on the neural unit. Pieces are loaded in stages so the prompt reader (1.1 GB)
// and the drawing pieces (2.0 GB) are never resident together. The drawing pieces, decoder and guide
// reader hold one graph per picture size ("s<W>x<H>", sharing their weights); only the size in use is loaded.
class Pipeline {
public:
    bool init(const std::string& modelDir, const std::string& libDir, std::string* err);
    GenResult generate(const GenRequest& req, const std::function<void(int, int)>& progress);
    // 4x on the neural unit (XLSR, in overlapping 512x512 tiles), any picture size.
    std::vector<uint32_t> upscale(const std::vector<uint32_t>& argb, int w, int h, std::string* err, double* ms);
    void release();
    // Memory use: 0 fastest (everything stays loaded), 1 normal (prompt encoder and drawing pieces take
    // turns), 2 low (only drawing pieces 0-1 stay loaded, 2-3 are swapped in for each step),
    // 3 lowest (one drawing piece loaded at a time).
    void setMemoryMode(int m) { mode_ = m; if (m != 0) release(); }
    static std::vector<float> sigmas(int steps);
    // Picture sizes the installed pieces were built for, "WxH" (just 512x512 for single-size files).
    const std::vector<std::string>& sizes() const { return sizes_; }
    // True when the installed pieces can follow a guide picture.
    bool canGuide() const { return guideFile_; }

private:
    bool useSize(int w, int h, std::string* err);
    bool loadDraw(std::string* err);
    bool loadPart(int k, std::string* err);
    int resident() const { return mode_ <= 1 ? 4 : mode_ == 2 ? 2 : 0; }
    bool runText(const GenRequest& req, std::vector<float>* ctx, std::string* err);
    bool runGuide(const GenRequest& req, std::vector<float>* cemb, std::string* err);
    int mode_ = 1;
    uint64_t groupSpill_ = 0;  // largest scratch need among the decoder and drawing pieces at this size
    std::string dir_;
    // Qwen token table, already divided by the residual scale: fp16 rows (qwen_embed_f16.raw), or 8-bit rows
    // with one float scale per row (qwen_embed_q8.raw, half the download; preferred when present)
    const uint16_t* qwenTable_ = nullptr;
    const int8_t* qwenQ8_ = nullptr;
    const float* qwenScale_ = nullptr;
    const uint16_t* t5Table_ = nullptr;
    size_t qwenBytes_ = 0, t5Bytes_ = 0;
    std::vector<std::string> sizes_;
    bool guideFile_ = false;
    int w_ = 0, h_ = 0;               // size the loaded pieces are for
    std::string partG_, decG_, guideG_;  // graph to use in each file at that size ("" = the only one)
    QnnGraph text_, part_[4], dec_;
};
