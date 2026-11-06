#include "fibo.h"

#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <memory>
#include <random>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "fibo", __VA_ARGS__)

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

constexpr _Float16 NEG = (_Float16)-30000.f;

// The drawing model's rotation: per row, 16 / 24 / 24 angle pairs for the three position axes (theta 256), each
// angle written twice (the two halves of a head are rotated together).
void drawRope(const float* ids3, int n, _Float16* P, _Float16* Q) {
    const int dims[3] = {32, 48, 48};
    for (int r = 0; r < n; r++) {
        int k = 0;
        for (int a = 0; a < 3; a++)
            for (int i = 0; i < dims[a] / 2; i++, k++) {
                double ang = ids3[r * 3 + a] * std::pow(256.0, -2.0 * i / dims[a]);
                P[r * 128 + k] = P[r * 128 + 64 + k] = (_Float16)std::cos(ang);
                Q[r * 128 + k] = Q[r * 128 + 64 + k] = (_Float16)std::sin(ang);
            }
    }
}

// The block streams xg<k>, xi<k>, xr<k> (k = 1..46) alternate between two buffers each: block k reads <k % 2> and
// writes <(k + 1) % 2>. Every other name keeps its own buffer.
std::string streamKey(const std::string& name) {
    if (name.size() < 3 || name[0] != 'x' || (name[1] != 'g' && name[1] != 'i' && name[1] != 'r')) return name;
    for (size_t i = 2; i < name.size(); i++) if (name[i] < '0' || name[i] > '9') return name;
    return name.substr(0, 2) + std::to_string(atoi(name.c_str() + 2) % 2);
}

}  // namespace

Fibo::~Fibo() {
    release();
    if (table_) munmap((void*)table_, tableBytes_);
}

void Fibo::release() {
    for (auto& b : bufs_) g_rpc.free(b.second.p);
    bufs_.clear();
}

Shm& Fibo::buf(const std::string& key, size_t bytes, std::string* err) {
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

bool Fibo::bindAll(QnnCtx& c, std::string* err, const std::string& prefix) {
    for (auto& g : c.graphs())
        for (int io = 0; io < 2; io++) {
            auto& names = io ? g.outNames : g.inNames;
            auto& bytes = io ? g.outBytes : g.inBytes;
            for (size_t i = 0; i < names.size(); i++) {
                Shm& s = buf(prefix + streamKey(names[i]), bytes[i], err);
                if (!err->empty()) return false;
                if (!c.bind(g, io == 0, i, s, err)) return false;
            }
        }
    return true;
}

// Loads one file, runs its first graph, and frees it.
bool Fibo::runFile(const std::string& name, std::string* err, const std::string& prefix) {
    QnnCtx c;
    if (!c.load(dir_ + "/" + name, err) || !bindAll(c, err, prefix)) return false;
    if (c.graphs().empty()) { *err = name + " has no graph"; return false; }
    return c.exec(c.graphs()[0], err);
}

// Runs files in order while the next ones load on a helper thread.
bool Fibo::pipeline(const std::vector<std::string>& paths, const std::function<bool(QnnCtx&, std::string*)>& run,
                    double* waitMs, std::string* err) {
    const int n = (int)paths.size();
    std::vector<std::unique_ptr<QnnCtx>> ctxs(n);
    std::vector<std::future<bool>> loads(n);
    std::vector<std::string> errs(n);
    auto start = [&](int k) {
        loads[k] = std::async(std::launch::async, [&, k] {
            ctxs[k].reset(new QnnCtx());
            if (!ctxs[k]->load(paths[k], &errs[k])) return false;
            std::lock_guard<std::mutex> lk(bufMutex_);
            return bindAll(*ctxs[k], &errs[k]);
        });
    };
    int next = 0;
    for (; next < std::min(n, ahead_); next++) start(next);
    bool ok = true;
    for (int k = 0; k < n; k++) {
        double tw = nowMs();
        if (!loads[k].get()) { *err = errs[k]; ok = false; break; }
        if (waitMs) *waitMs += nowMs() - tw;
        if (next < n) start(next++);
        if (!run(*ctxs[k], err)) { ok = false; break; }
        ctxs[k].reset();
    }
    for (int k = 0; k < n; k++) if (loads[k].valid()) loads[k].wait();
    return ok;
}

bool Fibo::init(const std::string& dir, const std::string& libDir, std::string* err) {
    if (table_) return true;
    dir_ = dir;
    if (const char* a = getenv("FIBO_AHEAD")) ahead_ = std::max(1, atoi(a));
    if (!QnnBackend::get().open(libDir, err)) return false;
    if (!g_rpc.open()) { *err = "libcdsprpc.so not available"; return false; }
    for (const char* f : {"fvis.bin", "fr0.bin", "fprep.bin", "fd0.bin", "fd45.bin", "fenc.bin", "fdec.bin"})
        if (access((dir + "/" + f).c_str(), R_OK) != 0) { *err = std::string("Fibo file missing: ") + f; return false; }
    int fd = open((dir + "/fibo_table_q8.raw").c_str(), O_RDONLY);
    if (fd < 0) { *err = "no fibo_table_q8.raw in " + dir; return false; }
    tableBytes_ = (size_t)lseek(fd, 0, SEEK_END);
    void* m = mmap(nullptr, tableBytes_, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { *err = "cannot map the word table"; return false; }
    table_ = (const int8_t*)m;
    scale_ = (const float*)(table_ + tableBytes_ / (D + 4) * D);
    return true;
}

bool Fibo::generate(const std::vector<int32_t>& ids, uint64_t seed, const uint8_t* editRgb, FiboResult* r,
                    std::string* err, const std::function<void(int, int)>& progress) {
    // noise levels (sigmas.txt, else the edit-turbo's own four), then 0
    std::vector<double> sig;
    if (FILE* f = fopen((dir_ + "/sigmas.txt").c_str(), "r")) {
        double v;
        while (fscanf(f, "%lf", &v) == 1) if (v > 0 && v <= 1) sig.push_back(v);
        fclose(f);
    }
    if (sig.empty()) sig = {1.0, 0.904531, 0.759511, 0.512844};
    const int STEPS = (int)sig.size();
    sig.push_back(0);
    if (progress) progress(0, STEPS);

    const int n = (int)ids.size();
    if (n > T) { *err = "edit instruction too long (more than 704 words with the picture)"; return false; }
    if (!editRgb) { *err = "Fibo needs a picture to edit"; return false; }
    int img = -1;   // first of the picture's 576 <|image_pad|> rows
    for (int i = 0; i < n && img < 0; i++) if (ids[i] == 151655) img = i;
    if (img < 0 || img + NV > n || ids[img + NV - 1] != 151655) { *err = "edit prompt needs 576 <|image_pad|> words"; return false; }
    const int hh = S / 16, ww = S / 16, N = hh * ww;

    // ---- the picture: the reader's vision tower, and the tiny encoder (the drawing model's copy of it)
    double t0 = nowMs();
    {
        Shm& px = buf("vpx", (size_t)VP * 768 * 2, err);
        Shm& ei = buf("eimg", (size_t)3 * S * S * 2, err);
        if (!err->empty()) return false;
        auto* v = (_Float16*)px.p;   // patches in Qwen's order: 2x2 groups row-major, then (channel, y, x)
        for (int p = 0; p < VP; p++) {
            int g = p / 4, by = g / 24, bx = g % 24, my = (p / 2) % 2, mx = p % 2;
            int y0 = (by * 2 + my) * 16, x0 = (bx * 2 + mx) * 16;
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++)
                        v[(size_t)p * 768 + c * 256 + y * 16 + x] =
                            (_Float16)(editRgb[((size_t)(y0 + y) * S + x0 + x) * 3 + c] / 127.5f - 1.f);
        }
        auto* e = (_Float16*)ei.p;
        for (size_t q = 0; q < (size_t)S * S; q++)
            for (int c = 0; c < 3; c++) e[c * S * S + q] = (_Float16)(editRgb[q * 3 + c] / 255.f);
        if (!runFile("fvis.bin", err) || !runFile("fenc.bin", err)) return false;
    }
    r->msVision = nowMs() - t0;

    // ---- reader (Qwen3-VL-4B's language half): the words, with the vision tower's rows in the picture's place
    t0 = nowMs();
    {
        buf("fx0", (size_t)T * D * 2, err); buf("fp", (size_t)T * HD * 2, err); buf("fq", (size_t)T * HD * 2, err);
        buf("fmask", (size_t)T * T * 2, err);
        for (int i = 0; i < 3; i++) buf("fds" + std::to_string(i), (size_t)T * D * 2, err);
        if (!err->empty()) return false;
        auto* x = (_Float16*)bufs_["fx0"].p;
        for (int row = 0; row < T; row++) {
            if (row >= img && row < img + NV) {
                memcpy(x + (size_t)row * D, (const _Float16*)bufs_["vm"].p + (size_t)(row - img) * D, D * 2);
                continue;
            }
            int id = row < n ? ids[row] : 151643;
            const int8_t* q = table_ + (size_t)id * D;
            float s = scale_[id];
            for (int k = 0; k < D; k++) x[(size_t)row * D + k] = (_Float16)(q[k] * s);
        }
        for (int i = 0; i < 3; i++) {   // the vision tower's deep features, added at the picture's rows
            auto* d = (_Float16*)bufs_["fds" + std::to_string(i)].p;
            memset(d, 0, (size_t)T * D * 2);
            memcpy(d + (size_t)img * D, bufs_["vd" + std::to_string(i)].p, (size_t)NV * D * 2);
        }
        // positions: text counts up; the picture's rows sit at (img, img + y, img + x) on a 24x24 grid, the text after
        // it continues at img + 24, and the angle pairs alternate axes
        auto* P = (_Float16*)bufs_["fp"].p; auto* Q = (_Float16*)bufs_["fq"].p;
        for (int p = 0; p < T; p++) {
            double pos[3] = {(double)p, (double)p, (double)p};
            if (p >= img && p < img + NV) {
                pos[0] = img; pos[1] = img + (p - img) / 24; pos[2] = img + (p - img) % 24;
            } else if (p >= img + NV) {
                pos[0] = pos[1] = pos[2] = img + 24 + (p - img - NV);
            }
            for (int i = 0; i < 64; i++) {
                int axis = i < 60 ? i % 3 : 0;
                double a = pos[axis] * std::pow(5e6, -2.0 * i / 128);
                P[p * 128 + i] = P[p * 128 + 64 + i] = (_Float16)std::cos(a);
                Q[p * 128 + i] = Q[p * 128 + 64 + i] = (_Float16)std::sin(a);
            }
        }
        auto* M = (_Float16*)bufs_["fmask"].p;
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++) M[(size_t)i * T + j] = (j <= i && (j < n || j == i)) ? (_Float16)0.f : NEG;
        std::vector<std::string> ps;
        for (int k = 0; k < 4; k++) ps.push_back(dir_ + "/fr" + std::to_string(k) + ".bin");
        auto run = [](QnnCtx& c, std::string* e) {
            QnnCtx::Graph* g = c.graph("t704");
            if (!g) { *e = "reader graph t704 missing"; return false; }
            return c.exec(*g, e);
        };
        if (!pipeline(ps, run, nullptr, err)) return false;
        // the reader's own buffers are no longer needed
        for (const char* k : {"fx0", "fx1", "fx2", "fx3", "fmask", "fds0", "fds1", "fds2", "vpx", "vm", "vd0", "vd1", "vd2"}) {
            auto it = bufs_.find(k);
            if (it != bufs_.end()) { g_rpc.free(it->second.p); bufs_.erase(it); }
        }
    }
    r->msReader = nowMs() - t0;

    // ---- once per edit: the 320 gist rows and the injection blocks' prompt keys/values
    t0 = nowMs();
    std::vector<_Float16> xg0;
    {
        Shm& kb = buf("kb", (size_t)T * 2, err);
        if (!err->empty()) return false;
        auto* b = (_Float16*)kb.p;
        for (int j = 0; j < T; j++) b[j] = j < n ? (_Float16)0.f : NEG;
        if (!runFile("fprep.bin", err)) return false;
        Shm& g = bufs_["xg0"];   // "xg2" lands in stream buffer xg0; each step overwrites it, so keep a copy
        xg0.assign((const _Float16*)g.p, (const _Float16*)g.p + g.bytes / 2);
        for (auto it = bufs_.begin(); it != bufs_.end();) {   // the taps were read; free them
            if (it->first.rfind("ft", 0) == 0) { g_rpc.free(it->second.p); it = bufs_.erase(it); } else ++it;
        }
    }
    r->msPrep = nowMs() - t0;

    // ---- drawing: 46 blocks per step, each its own file, loaded in turn
    t0 = nowMs();
    buf("pg", (size_t)G * HD * 2, err); buf("qg", (size_t)G * HD * 2, err);
    buf("pi", (size_t)N * HD * 2, err); buf("qi", (size_t)N * HD * 2, err);
    buf("pr", (size_t)N * HD * 2, err); buf("qr", (size_t)N * HD * 2, err);
    buf("lat", (size_t)N * C * 2, err); buf("ts", 256 * 2, err);
    if (!err->empty()) return false;
    {
        // positions: gist (1..320, 0, 0); the picture being drawn (513, y, x); the edit picture (514, y, x)
        std::vector<float> g3((size_t)G * 3, 0.f), i3((size_t)N * 3), r3((size_t)N * 3);
        for (int k = 0; k < G; k++) g3[k * 3] = (float)(k + 1);
        for (int y = 0; y < hh; y++)
            for (int x = 0; x < ww; x++) {
                size_t p = (size_t)y * ww + x;
                i3[p * 3] = 513; r3[p * 3] = 514;
                i3[p * 3 + 1] = r3[p * 3 + 1] = (float)y;
                i3[p * 3 + 2] = r3[p * 3 + 2] = (float)x;
            }
        drawRope(g3.data(), G, (_Float16*)bufs_["pg"].p, (_Float16*)bufs_["qg"].p);
        drawRope(i3.data(), N, (_Float16*)bufs_["pi"].p, (_Float16*)bufs_["qi"].p);
        drawRope(r3.data(), N, (_Float16*)bufs_["pr"].p, (_Float16*)bufs_["qr"].p);
    }
    std::vector<float> z((size_t)N * C);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : z) v = nd(rng);
    std::vector<std::string> ps;
    for (int k = 0; k < 46; k++) ps.push_back(dir_ + "/fd" + std::to_string(k) + ".bin");
    for (int i = 0; i < STEPS; i++) {
        if (progress) progress(i + 1, STEPS);
        auto* lat = (_Float16*)bufs_["lat"].p;
        for (size_t k = 0; k < z.size(); k++) lat[k] = (_Float16)z[k];
        auto* ts = (_Float16*)bufs_["ts"].p;
        for (int k = 0; k < 128; k++) {
            double a = sig[i] * 1000.0 * std::exp(-std::log(10000.0) * k / 128.0);
            ts[k] = (_Float16)std::cos(a); ts[128 + k] = (_Float16)std::sin(a);
        }
        if (i > 0) memcpy(bufs_["xg0"].p, xg0.data(), xg0.size() * 2);
        auto run = [](QnnCtx& c, std::string* e) { return c.exec(c.graphs()[0], e); };
        if (!pipeline(ps, run, &r->msSwapLoad, err)) return false;
        LOGI("step %d of %d done", i + 1, STEPS);
        auto* vel = (const _Float16*)bufs_["vel"].p;
        for (size_t k = 0; k < z.size(); k++) z[k] += (float)(sig[i + 1] - sig[i]) * (float)vel[k];
    }
    r->msSteps = nowMs() - t0;

    // ---- decode (tiny decoder, 768x768)
    t0 = nowMs();
    {
        QnnCtx c;
        if (!c.load(dir_ + "/fdec.bin", err) || !bindAll(c, err, "dec_")) return false;
        auto* lat = (_Float16*)bufs_["dec_lat"].p;   // (128, 48, 48) from rows (N, 128)
        for (int p = 0; p < N; p++)
            for (int ch = 0; ch < C; ch++) lat[(size_t)ch * N + p] = (_Float16)z[(size_t)p * C + ch];
        QnnCtx::Graph* g = c.graph("s768x768");
        if (!g || !c.exec(*g, err)) { if (err->empty()) *err = "no decoder graph s768x768"; return false; }
        auto* rgb = (const _Float16*)bufs_["dec_rgb"].p;
        const size_t NP = (size_t)S * S;
        r->argb.resize(NP);
        auto u8 = [](float v) { return (uint32_t)std::lround(std::min(1.f, std::max(0.f, v)) * 255.f); };
        for (size_t p = 0; p < NP; p++)
            r->argb[p] = 0xFF000000u | (u8((float)rgb[p]) << 16) | (u8((float)rgb[NP + p]) << 8) | u8((float)rgb[2 * NP + p]);
    }
    r->msDecode = nowMs() - t0;
    LOGI("times ms: vision %.0f reader %.0f prep %.0f steps %.0f (waiting for loads %.0f) decode %.0f", r->msVision, r->msReader, r->msPrep, r->msSteps, r->msSwapLoad, r->msDecode);
    release();   // ~1 GB of shared buffers; the next edit sets them up again
    return true;
}
