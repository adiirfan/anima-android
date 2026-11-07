#include "wan.h"
#include "QnnTypeMacros.hpp"

#include <android/log.h>
#include <dlfcn.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <memory>
#include <random>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "wan", __VA_ARGS__)

namespace {

double nowMs() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Rpc {
    void* (*alloc)(int, uint32_t, int) = nullptr;
    void (*free)(void*) = nullptr;
    int (*toFd)(void*) = nullptr;
    bool open() {
        if (alloc) return true;
        void* l = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
        if (!l) return false;
        alloc = (void* (*)(int, uint32_t, int))dlsym(l, "rpcmem_alloc");
        free = (void (*)(void*))dlsym(l, "rpcmem_free");
        toFd = (int (*)(void*))dlsym(l, "rpcmem_to_fd");
        return alloc && free && toFd;
    }
} g_rpc;

// The block stream x<k> (k = 0..30) alternates between two buffers: block k reads x<k % 2>, writes x<(k + 1) % 2>.
std::string streamKey(const std::string& name) {
    if (name.size() < 2 || name[0] != 'x') return name;
    for (size_t i = 1; i < name.size(); i++) if (name[i] < '0' || name[i] > '9') return name;
    return "x" + std::to_string(atoi(name.c_str() + 1) % 2);
}

// Wan's rotation: the head's 64 angle pairs split over (frame, y, x) as 22 + 21 + 21, theta 10000, each angle written
// for both halves of the head (the blocks' q/k features are reordered to that layout). Rows ordered (frame, y, x).
void rope(int f, int h, int w, _Float16* P, _Float16* Q) {
    const int dims[3] = {44, 42, 42};
    int r = 0;
    for (int a = 0; a < f; a++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++, r++) {
                const double pos[3] = {(double)a, (double)y, (double)x};
                int k = 0;
                for (int ax = 0; ax < 3; ax++)
                    for (int i = 0; i < dims[ax] / 2; i++, k++) {
                        double ang = pos[ax] * std::pow(10000.0, -2.0 * i / dims[ax]);
                        P[(size_t)r * 128 + k] = P[(size_t)r * 128 + 64 + k] = (_Float16)std::cos(ang);
                        Q[(size_t)r * 128 + k] = Q[(size_t)r * 128 + 64 + k] = (_Float16)std::sin(ang);
                    }
            }
}

// Wan's timestep embedding input: cos then sin of t * 10000^(-k / 128), t on the 0..1000 scale.
void tsRow(double t, _Float16* o) {
    for (int k = 0; k < 128; k++) {
        double a = t * std::pow(10000.0, -(double)k / 128.0);
        o[k] = (_Float16)std::cos(a); o[128 + k] = (_Float16)std::sin(a);
    }
}

// true when a 4-D picture tensor sits at the graph's edge channels-last (N, H, W, C) rather than (N, C, H, W)
bool channelsLast(const QnnCtx::Graph& g, bool input, const std::string& name, uint32_t channels) {
    auto& names = input ? g.inNames : g.outNames;
    auto& ts = input ? g.in : g.out;
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] != name) continue;
        const Qnn_Tensor_t& t = ts[i];
        uint32_t rank = QNN_TENSOR_GET_RANK(t);
        const uint32_t* d = QNN_TENSOR_GET_DIMENSIONS(t);
        return rank == 4 && d[3] == channels && d[1] != channels;
    }
    return false;
}

}  // namespace

// The video model works on latents rescaled per channel, (latent - mean) / std; the LightTAE encoder and decoder use
// the unscaled ones. ComfyUI's Wan22 latent format; without it every frame after the first breaks into blocks.
static const float kLatMean[48] = {-0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f, -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.1570f, -0.0098f, 0.0375f, -0.1825f, -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f, -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.1230f, -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.0520f, 0.3748f, 0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f};
static const float kLatStd[48] = {0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f, 0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f, 0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f, 0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f, 0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f, 0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f};

// Debugging: when the model folder holds a "dump" folder, the start latents, the prompt rows, the first step's
// velocity and the final latents are written there as raw files (compared with sim_clip.py on the computer).
static void wanDump(const std::string& dir, const char* name, const void* p, size_t bytes) {
    if (access((dir + "/dump").c_str(), F_OK) != 0) return;
    if (FILE* f = fopen((dir + "/dump/" + name).c_str(), "wb")) { fwrite(p, 1, bytes, f); fclose(f); }
}

Wan::~Wan() { release(); }

void Wan::release() {
    for (auto& b : bufs_) g_rpc.free(b.second.p);
    bufs_.clear();
}

Shm& Wan::buf(const std::string& key, size_t bytes, std::string* err) {
    auto it = bufs_.find(key);
    if (it != bufs_.end()) {
        if (it->second.bytes != bytes) *err = "buffer " + key + " size mismatch";
        return it->second;
    }
    Shm s;
    s.p = g_rpc.alloc(25, 1, (int)bytes);
    if (!s.p) { *err = "rpcmem_alloc failed for " + key; return bufs_[key] = s; }
    s.bytes = bytes; s.fd = g_rpc.toFd(s.p);
    memset(s.p, 0, bytes);
    return bufs_[key] = s;
}

bool Wan::bindAll(QnnCtx& c, std::string* err) {
    for (auto& g : c.graphs())
        for (int io = 0; io < 2; io++) {
            auto& names = io ? g.outNames : g.inNames;
            auto& bytes = io ? g.outBytes : g.inBytes;
            for (size_t i = 0; i < names.size(); i++) {
                Shm& s = buf(streamKey(names[i]), bytes[i], err);
                if (!err->empty()) return false;
                if (!c.bind(g, io == 0, i, s, err)) return false;
            }
        }
    return true;
}

bool Wan::runFile(const std::string& name, std::string* err) {
    QnnCtx c;
    if (!c.load(dir_ + "/" + name, err) || !bindAll(c, err)) return false;
    if (c.graphs().empty()) { *err = name + " has no graph"; return false; }
    return c.exec(c.graphs()[0], err);
}

bool Wan::init(const std::string& dir, const std::string& libDir, std::string* err) {
    if (ready_) return true;
    dir_ = dir;
    if (const char* a = getenv("WAN_AHEAD")) ahead_ = std::max(1, atoi(a));
    if (!QnnBackend::get().open(libDir, err)) return false;
    if (!g_rpc.open()) { *err = "libcdsprpc.so not available"; return false; }
    for (const char* f : {"enc.bin", "prep.bin", "time.bin", "emb.bin", "w0.bin", "w29.bin", "head.bin", "dec.bin", "sigmas.txt"})
        if (access((dir + "/" + f).c_str(), R_OK) != 0) { *err = std::string("Wan file missing: ") + f; return false; }
    LOGI("Wan 2.2 ready");
    ready_ = true;
    return true;
}

bool Wan::generate(const float* rows, int n, const uint8_t* rgb, uint64_t seed, WanResult* r, std::string* err,
                   const std::function<void(int, int)>& progress) {
    std::vector<double> sig;
    if (FILE* f = fopen((dir_ + "/sigmas.txt").c_str(), "r")) {
        double v;
        while (fscanf(f, "%lf", &v) == 1) if (v > 0 && v <= 1) sig.push_back(v);
        fclose(f);
    }
    if (sig.empty()) { *err = "no noise levels in sigmas.txt"; return false; }
    const int STEPS = (int)sig.size();
    sig.push_back(0);
    if (n < 1 || n > T) { *err = "prompt too long (more than 512 words)"; return false; }
    if (progress) progress(0, STEPS);
    const size_t PLANE = (size_t)LH * LH, LAT = (size_t)C * F * PLANE;
    auto zat = [&](int c, int f) { return (size_t)c * F * PLANE + (size_t)f * PLANE; };   // z laid out (C, F, H, W)

    // ---- the start picture -> its latent frame (LightTAE encoder)
    double t0 = nowMs();
    std::vector<float> z0(C * PLANE);
    {
        QnnCtx c;
        if (!c.load(dir_ + "/enc.bin", err) || !bindAll(c, err)) return false;
        const bool nhwc = channelsLast(c.graphs()[0], true, "img", 3);
        auto* im = (_Float16*)bufs_["img"].p;
        const size_t NP = (size_t)S * S;
        for (size_t p = 0; p < NP; p++)
            for (int ch = 0; ch < 3; ch++)
                im[nhwc ? p * 3 + ch : ch * NP + p] = (_Float16)(rgb[p * 3 + ch] / 255.f);
        if (!c.exec(c.graphs()[0], err)) return false;
        const bool znhwc = channelsLast(c.graphs()[0], false, "z0", C) && !channelsLast(c.graphs()[0], false, "z0", LH);
        auto* zz = (const _Float16*)bufs_["z0"].p;
        for (int ch = 0; ch < C; ch++)
            for (size_t p = 0; p < PLANE; p++)
                z0[ch * PLANE + p] = ((float)zz[znhwc ? p * C + ch : ch * PLANE + p] - kLatMean[ch]) / kLatStd[ch];
        LOGI("start picture encoded (%s)", nhwc ? "channels last" : "channels first");
    }
    // ---- the prompt -> every block's keys/values (kept for all steps)
    {
        Shm& cx = buf("ctx", (size_t)T * TXT * 2, err);
        if (!err->empty()) return false;
        auto* x = (_Float16*)cx.p;
        memset(x, 0, (size_t)T * TXT * 2);   // ComfyUI zeroes the rows after the words
        for (size_t k = 0; k < (size_t)n * TXT; k++) x[k] = (_Float16)rows[k];
        if (!runFile("prep.bin", err)) return false;
        auto it = bufs_.find("ctx");
        g_rpc.free(it->second.p); bufs_.erase(it);
    }
    // ---- rotation tables, shared by every block and step
    buf("pw", (size_t)L * 128 * 2, err); buf("qw", (size_t)L * 128 * 2, err);
    if (!err->empty()) return false;
    rope(F, PH, PH, (_Float16*)bufs_["pw"].p, (_Float16*)bufs_["qw"].p);
    r->msPrep = nowMs() - t0;

    // ---- drawing: frame 0 is the start picture (timestep 0) throughout, the 4 frames after it start as noise
    t0 = nowMs();
    std::vector<float> z(LAT);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int ch = 0; ch < C; ch++)
        for (int f = 0; f < F; f++)
            for (size_t p = 0; p < PLANE; p++) z[zat(ch, f) + p] = f == 0 ? z0[ch * PLANE + p] : nd(rng);
    wanDump(dir_, "zinit.f32", z.data(), z.size() * 4);
    wanDump(dir_, "rows.f32", rows, (size_t)n * TXT * 4);
    std::vector<std::string> paths;
    for (int k = 0; k < NB; k++) paths.push_back(dir_ + "/w" + std::to_string(k) + ".bin");
    for (int i = 0; i < STEPS; i++) {
        if (progress) progress(i + 1, STEPS);
        // the step's modulation rows: e0 of the start frame's rows at t = 0, the rest at the step's t
        {
            Shm& ts = buf("ts", 2 * 256 * 2, err);
            if (!err->empty()) return false;
            tsRow(0.0, (_Float16*)ts.p);
            tsRow(sig[i] * 1000.0, (_Float16*)ts.p + 256);
            if (!runFile("time.bin", err)) return false;
            auto* e = (const _Float16*)bufs_["e"].p;     // (2, D)
            auto* e0 = (const _Float16*)bufs_["e0"].p;   // (2, 6, D)
            Shm& er = buf("er", (size_t)L * D * 2, err);
            for (int j = 0; j < 6; j++) buf("m" + std::to_string(j), (size_t)L * D * 2, err);
            if (!err->empty()) return false;
            std::vector<_Float16> mrow[2][6];
            for (int g = 0; g < 2; g++)
                for (int j = 0; j < 6; j++) {
                    mrow[g][j].resize(D);
                    for (int d = 0; d < D; d++)
                        mrow[g][j][d] = (_Float16)((float)e0[((size_t)g * 6 + j) * D + d] + (j == 1 || j == 4 ? 1.f : 0.f));
                }
            for (int row = 0; row < L; row++) {
                const int g = row < N0 ? 0 : 1;
                for (int j = 0; j < 6; j++) memcpy((_Float16*)bufs_["m" + std::to_string(j)].p + (size_t)row * D, mrow[g][j].data(), D * 2);
                memcpy((_Float16*)er.p + (size_t)row * D, e + (size_t)g * D, D * 2);
            }
        }
        // the latent's 2x2 patches -> the stream x0
        {
            Shm& pt = buf("patches", (size_t)L * C * 4 * 2, err);
            if (!err->empty()) return false;
            auto* q = (_Float16*)pt.p;
            for (int f = 0; f < F; f++)
                for (int py = 0; py < PH; py++)
                    for (int px = 0; px < PH; px++) {
                        size_t rr = (size_t)f * N0 + py * PH + px;
                        for (int ch = 0; ch < C; ch++)
                            for (int dy = 0; dy < 2; dy++)
                                for (int dx = 0; dx < 2; dx++)
                                    q[rr * C * 4 + ch * 4 + dy * 2 + dx] = (_Float16)z[zat(ch, f) + (size_t)(2 * py + dy) * LH + 2 * px + dx];
                    }
            if (!runFile("emb.bin", err)) return false;
        }
        // the 30 blocks, loaded ahead on helper threads
        std::vector<std::unique_ptr<QnnCtx>> ctxs(NB);
        std::vector<std::future<bool>> loads(NB);
        std::vector<std::string> errs(NB);
        auto start = [&](int k) {
            loads[k] = std::async(std::launch::async, [&, k] {
                ctxs[k].reset(new QnnCtx());
                return ctxs[k]->load(paths[k], &errs[k]);
            });
        };
        int next = 0;
        for (; next < std::min(NB, ahead_); next++) start(next);
        bool ok = true;
        for (int k = 0; k < NB && ok; k++) {
            double tw = nowMs();
            if (!loads[k].get()) { *err = errs[k]; ok = false; break; }
            r->msLoadWait += nowMs() - tw;
            if (next < NB) start(next++);
            if (!bindAll(*ctxs[k], err) || !ctxs[k]->exec(ctxs[k]->graphs()[0], err)) ok = false;
            ctxs[k].reset();
        }
        for (int k = 0; k < NB; k++) if (loads[k].valid()) loads[k].wait();
        if (!ok) return false;
        if (!runFile("head.bin", err)) return false;
        // velocity rows (dy, dx, channel) -> one flow step; the start frame stays the clean picture
        auto* v = (const _Float16*)bufs_["vel"].p;
        const float dt = (float)(sig[i + 1] - sig[i]);
        if (i == 0) wanDump(dir_, "vel0.f16", v, bufs_["vel"].bytes);
        for (int f = 1; f < F; f++)
            for (int py = 0; py < PH; py++)
                for (int px = 0; px < PH; px++) {
                    size_t rr = (size_t)f * N0 + py * PH + px;
                    for (int dy = 0; dy < 2; dy++)
                        for (int dx = 0; dx < 2; dx++)
                            for (int ch = 0; ch < C; ch++)
                                z[zat(ch, f) + (size_t)(2 * py + dy) * LH + 2 * px + dx] += dt * (float)v[rr * C * 4 + (dy * 2 + dx) * C + ch];
                }
        LOGI("step %d of %d done", i + 1, STEPS);
    }
    r->msSteps = nowMs() - t0;
    wanDump(dir_, "zfinal.f32", z.data(), z.size() * 4);
    for (const char* k : {"m0", "m1", "m2", "m3", "m4", "m5", "er", "patches", "pw", "qw"}) {   // drawing's buffers
        auto it = bufs_.find(k);
        if (it != bufs_.end()) { g_rpc.free(it->second.p); bufs_.erase(it); }
    }

    // ---- decode: one latent frame at a time, 4 frames each; the first 3 frames are dropped (ComfyUI does too)
    t0 = nowMs();
    if (progress) progress(STEPS + 1, STEPS);
    // back from the model's scale to the decoder's
    for (int ch = 0; ch < C; ch++)
        for (size_t k = 0; k < (size_t)F * PLANE; k++) z[zat(ch, 0) + k] = z[zat(ch, 0) + k] * kLatStd[ch] + kLatMean[ch];
    {
        QnnCtx c;
        if (!c.load(dir_ + "/dec.bin", err) || !bindAll(c, err)) return false;
        QnnCtx::Graph& g = c.graphs()[0];
        const bool lnhwc = channelsLast(g, true, "lat", C) && !channelsLast(g, true, "lat", LH);
        const bool fnhwc = channelsLast(g, false, "frames", 3);
        for (int j = 0; j < 9; j++) memset(bufs_["dm" + std::to_string(j)].p, 0, bufs_["dm" + std::to_string(j)].bytes);
        const size_t NP = (size_t)S * S;
        r->argb.clear(); r->argb.reserve((size_t)OUTF * NP);
        auto u8 = [](float x) { return (uint32_t)std::lround(std::min(1.f, std::max(0.f, x)) * 255.f); };
        int made = 0;
        for (int f = 0; f < F; f++) {
            auto* lat = (_Float16*)bufs_["lat"].p;
            for (int ch = 0; ch < C; ch++)
                for (size_t p = 0; p < PLANE; p++) lat[lnhwc ? p * C + ch : ch * PLANE + p] = (_Float16)z[zat(ch, f) + p];
            if (!c.exec(g, err)) return false;
            for (int j = 0; j < 9; j++) {
                Shm& a = bufs_["dn" + std::to_string(j)];
                memcpy(bufs_["dm" + std::to_string(j)].p, a.p, a.bytes);
            }
            auto* fr = (const _Float16*)bufs_["frames"].p;   // 4 frames
            for (int q = 0; q < 4; q++, made++) {
                if (made < 3) continue;
                const _Float16* b = fr + (size_t)q * 3 * NP;
                for (size_t p = 0; p < NP; p++) {
                    float R = fnhwc ? b[p * 3] : b[p], G = fnhwc ? b[p * 3 + 1] : b[NP + p], B = fnhwc ? b[p * 3 + 2] : b[2 * NP + p];
                    r->argb.push_back(0xFF000000u | (u8(R) << 16) | (u8(G) << 8) | u8(B));
                }
            }
        }
        r->frames = (int)(r->argb.size() / NP);
    }
    r->msDecode = nowMs() - t0;
    LOGI("times ms: prepare %.0f steps %.0f (waiting for loads %.0f) decode %.0f, %d frames", r->msPrep, r->msSteps,
         r->msLoadWait, r->msDecode, r->frames);
    release();
    return true;
}
