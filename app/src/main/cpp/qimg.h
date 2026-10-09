#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "gemma.h"   // QnnCtx, Shm

// Qwen-Image 2.1 (4-step turbo merged) on the neural unit, from qwen-poc/: prompt reader r0-3.bin (Qwen3-VL-8B
// language half, 128 word slots, loaded one piece at a time) + qwen_table_q8.raw; drawing model sp0-15.bin (4-bit,
// 2 blocks per file, each block with a graph "prefix_<j>" = the prompt's rows once per picture and "s<W>x<H>_<j>" =
// the picture's rows per step, sharing one copy of the weights; loaded in turn for the prompt pass and every step);
// dec.bin (taeqi2_1 decoder). Older layouts (part/pack/spack+ppack files) still load.
// Two downloads share the drawing model: Standard (the 8B reader r0-3.bin + qwen_table_q8.raw, picture editing) and
// Small (Qwen3-VL-2B + a translator to the 8B's features: s0.bin, s1.bin, proj.bin, qwen2b_table_q8.raw; prompts
// only). The 2B reads when it is the one installed (or QIMG_READER=2b with both present).
// Editing (768x768): vis.bin (the reader's vision tower) + enc.bin (taeqi2_1 encoder) for the picture; the reader
// files' graph "t704" and the sp files' graphs "eprefix_<j>" / "e768x768_<j>" (same weights as the plain graphs).
// All tensors live in shared memory, named after the graph tensors, so the pieces pass data without copies.
struct QimgResult {
    std::vector<uint32_t> argb;
    double msVision = 0, msReader = 0, msLoad = 0, msPrefix = 0, msSteps = 0, msDecode = 0, msSwapLoad = 0;
};

class QwenImage {
public:
    ~QwenImage();
    // drawDir: another folder holding the drawing model's files (sp<k>.bin), e.g. a fine-tune of the same shape;
    // empty or without sp0.bin = dir. The readers, word tables and decoder always come from dir.
    bool init(const std::string& dir, const std::string& libDir, std::string* err, const std::string& drawDir = "");
    // ids: the whole chat-template prompt (system turn included), at most 128 words. Editing: editRgb is the
    // 768x768 RGB picture to edit and ids carry its 576 <|image_pad|> words where the template puts the picture
    // ("<|im_start|>user\n<image1><|vision_start|>" + 576 x <|image_pad|> + "<|vision_end|>" + prompt ...).
    // progress(i, n): drawing step i of n is starting (0 of n while reading the prompt).
    bool generate(const std::vector<int32_t>& ids, uint64_t seed, int width, int height, QimgResult* r, std::string* err,
                  const uint8_t* editRgb = nullptr, const std::function<void(int, int)>& progress = nullptr);
    // Frees everything on the neural unit and the shared buffers (an edit's keep ~1.3 GB); the next generate()
    // loads again.
    void release();

private:
    static constexpr int T = 128, D = 4096, HD = 128, C = 64;
    static constexpr int TE = 704, NV = 576, VP = 2304;   // edit reader rows, the picture's reader rows, its patches
    Shm& buf(const std::string& key, size_t bytes, std::string* err);
    bool bindAll(QnnCtx& c, std::string* err);
    bool loadReader(std::string* err);
    bool anchorGroup(std::string* err);
    uint64_t groupSpill_ = 0;                    // shared working area of the context group (0: none)
    std::function<bool(const std::string&)> want_;   // graphs to bind (the others in a file get no memory)
    bool loadDrawing(std::string* err);
    void freeReader();
    bool pipeline(const std::vector<std::string>& paths, int ahead, const std::function<bool(QnnCtx&, std::string*)>& run,
                  double* waitMs, std::string* err);
    int ahead_ = 1;                              // files loading ahead (QIMG_AHEAD; 2 measured slower: the loads share the neural unit)
    std::vector<std::pair<QnnCtx*, QnnCtx::Graph*>> graphsFor(const std::string& kind);

    std::string dir_;
    std::string draw_;   // drawing model files (sp<k>.bin and the older pack layouts)
    QnnCtx reader_[4], part_[32], dec_;
    int nParts_ = 0;                             // drawing files found (part0.bin ... or pack0.bin ...)
    bool packed_ = false;                        // pack files: one block per graph
    bool split_ = false;                         // spack (steps, kept) + ppack (prompt pass, per picture)
    bool swap_ = false;                          // sp<k>.bin: per-step files loaded in turn every step
    bool readerLoaded_ = false, drawingLoaded_ = false;
    std::map<std::string, Shm> bufs_;
    std::mutex bufMutex_;                        // bufs_ is also used by the loading thread
    const int8_t* table_ = nullptr;
    const float* scale_ = nullptr;
    size_t tableBytes_ = 0;
    // the small (2B) reader, when its files are there: s0.bin, s1.bin, proj.bin + qwen2b_table_q8.raw (width 2048)
    static constexpr int DS = 2048;
    const int8_t* stable_ = nullptr;
    const float* sscale_ = nullptr;
    size_t stableBytes_ = 0;
};
