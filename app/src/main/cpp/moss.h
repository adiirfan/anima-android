#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "gemma.h"   // QnnCtx, Shm

// MOSS-Image (SFT) on the neural unit: draws a 768x768 picture from a prompt in 25 steps, each with the prompt and the
// empty prompt (guidance 2.5, block 22 left out of the empty-prompt pass for the first steps).
// The prompt is read beforehand by llama.cpp (Qwen 3.5 9B, see MossReader.kt): rows of 4096 numbers per word.
// Files in one folder: mtxt.bin (graph "txt": reader rows -> the model's prompt rows, once per prompt), mtime.bin
// (graph "time": the step's noise level -> the shared modulation), md0-41.bin (the 42 blocks, one per file, loaded in
// turn every step and run for both passes), fdec.bin (Flux.2's tiny decoder at 768, the same file as Fibo's),
// sigmas.txt (the 25 noise levels). Tensors live in shared memory named after the graph tensors, "c_" for the prompt
// pass and "u_" for the empty one; the block streams xi/xt<k> alternate between two buffers each.
struct MossResult {
    std::vector<uint32_t> argb;
    double msText = 0, msSteps = 0, msDecode = 0, msLoadWait = 0;
};

class Moss {
public:
    ~Moss();
    bool init(const std::string& dir, const std::string& libDir, std::string* err);
    // cond: nc rows of 4096 (the prompt), unc: nu rows (the empty prompt), both at most 256 rows; in the 8-step
    // build (sampler.txt "euler") unc is not used and may be empty.
    // progress(i, n): step i of n is starting (0 while preparing).
    bool generate(const float* cond, int nc, const float* unc, int nu, uint64_t seed, MossResult* r, std::string* err,
                  const std::function<void(int, int)>& progress = nullptr);
    void release();   // frees the shared buffers

private:
    static constexpr int T = 256, D = 3072, HD = 128, C = 128, TXT = 4096, NB = 42, S = 768;
    static constexpr int SLG_BLOCK = 22;
    static constexpr double SLG_END = 0.833333, CFG = 2.5;
    Shm& buf(const std::string& key, size_t bytes, std::string* err);
    bool bindAll(QnnCtx& c, std::string* err, const std::string& prefix);
    bool runFile(const std::string& name, std::string* err, const std::string& prefix);
    std::string dir_;
    int ahead_ = 2;   // block files loading ahead (MOSS_AHEAD)
    std::map<std::string, Shm> bufs_;
    bool ready_ = false;
    bool euler_ = false;   // sampler.txt says "euler": the 8-step add-on build, plain flow steps, prompt pass only
};
