#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "gemma.h"   // QnnCtx, Shm

// Wan 2.2 TI2V-5B Turbo on the neural unit: turns a 768x768 picture into a 17-frame clip (8 steps, no guidance).
// The prompt (how the picture should move) is read beforehand by llama.cpp (umT5-XXL, see WanReader.kt): rows of
// 4096 numbers per word, zero-padded to 512 here.
// Files in one folder (graph names in brackets):
//   enc.bin  [enc]   LightTAE encoder: the picture -> its latent frame z0 (48 x 48 x 48)
//   prep.bin [prep]  the reader rows -> every block's prompt keys/values ck<k>, cv<k> (once per clip)
//   time.bin [time]  the two rows' timestep (start frame: 0, the rest: the step's) -> e, e0
//   emb.bin  [emb]   the latent's 2x2 patches -> the stream x0 (2880 rows)
//   w0-29.bin [w<k>] the 30 blocks, loaded in turn every step
//   head.bin [head]  the last stream -> the velocity rows
//   dec.bin  [dec]   LightTAE decoder step: one latent frame + 9 memories -> 4 frames + new memories
//   sigmas.txt       the 8 noise levels
struct WanResult {
    std::vector<uint32_t> argb;   // frames * 768 * 768 ARGB, frame after frame
    int frames = 0;
    double msPrep = 0, msSteps = 0, msDecode = 0, msLoadWait = 0;
};

class Wan {
public:
    ~Wan();
    bool init(const std::string& dir, const std::string& libDir, std::string* err);
    // rows: n x 4096 (the prompt as read by the reader, n <= 512); rgb: the start picture, 768 x 768 x 3 bytes.
    // progress(i, n): step i of n is starting (0 while preparing, n + 1 while decoding).
    bool generate(const float* rows, int n, const uint8_t* rgb, uint64_t seed, WanResult* r, std::string* err,
                  const std::function<void(int, int)>& progress = nullptr);
    void release();

private:
    static constexpr int S = 768, LH = 48, C = 48, F = 5, PH = 24, D = 3072, T = 512, TXT = 4096, NB = 30;
    static constexpr int N0 = PH * PH, L = F * N0, OUTF = 17;
    Shm& buf(const std::string& key, size_t bytes, std::string* err);
    bool bindAll(QnnCtx& c, std::string* err);
    bool runFile(const std::string& name, std::string* err);
    std::string dir_;
    int ahead_ = 2;
    std::map<std::string, Shm> bufs_;
    bool ready_ = false;
};
