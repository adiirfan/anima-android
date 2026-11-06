#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "gemma.h"   // QnnCtx, Shm

// Fibo 2.0 edit-turbo on the neural unit: changes a 768x768 picture following a written instruction, in 4 steps.
// Files in one folder: fvis.bin (the reader's vision tower), fr0-3.bin (the reader, Qwen3-VL-4B's language half, 704
// word slots, graph "t704") + fibo_table_q8.raw (its word table), fprep.bin (once per edit: the reader's 15 taps ->
// the 320 "gist" rows and the five injection blocks' prompt keys/values), fd0-45.bin (the 46 drawing blocks, one per
// file, loaded in turn every step), fenc.bin / fdec.bin (Flux.2's tiny encoder and decoder at 768), sigmas.txt.
// All tensors live in shared memory named after the graph tensors, so the pieces pass data without copies; the
// block streams xg/xi/xr<k> alternate between two buffers each.
struct FiboResult {
    std::vector<uint32_t> argb;
    double msVision = 0, msReader = 0, msPrep = 0, msSteps = 0, msDecode = 0, msSwapLoad = 0;
};

class Fibo {
public:
    ~Fibo();
    bool init(const std::string& dir, const std::string& libDir, std::string* err);
    // ids: "<|im_start|>user\n" + the JSON edit request with its 576 <|image_pad|> words + "<|im_end|>\n
    // <|im_start|>assistant\n", at most 704 words. editRgb: the 768x768 RGB picture to change.
    // progress(i, n): step i of n is starting (0 while reading).
    bool generate(const std::vector<int32_t>& ids, uint64_t seed, const uint8_t* editRgb, FiboResult* r, std::string* err,
                  const std::function<void(int, int)>& progress = nullptr);
    void release();   // frees the neural unit and the shared buffers

private:
    static constexpr int T = 704, D = 2560, HD = 128, C = 128, NV = 576, VP = 2304, G = 320, S = 768;
    Shm& buf(const std::string& key, size_t bytes, std::string* err);
    bool bindAll(QnnCtx& c, std::string* err, const std::string& prefix = "");
    bool runFile(const std::string& name, std::string* err, const std::string& prefix = "");
    bool pipeline(const std::vector<std::string>& paths, const std::function<bool(QnnCtx&, std::string*)>& run,
                  double* waitMs, std::string* err);
    std::string dir_;
    int ahead_ = 1;   // files loading ahead (FIBO_AHEAD)
    std::map<std::string, Shm> bufs_;
    std::mutex bufMutex_;
    const int8_t* table_ = nullptr;
    const float* scale_ = nullptr;
    size_t tableBytes_ = 0;
};
