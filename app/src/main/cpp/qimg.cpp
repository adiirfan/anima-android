#include "qimg.h"

#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <random>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "qimg", __VA_ARGS__)

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

// Debugging: with QIMG_DUMP=<dir> set, named buffers are written there as raw fp16 files.
void dumpRaw(const char* name, const void* p, size_t bytes) {
    const char* d = getenv("QIMG_DUMP");
    if (!d || !p) return;
    FILE* f = fopen((std::string(d) + "/" + name + ".f16").c_str(), "wb");
    if (f) { fwrite(p, 1, bytes, f); fclose(f); }
}

// one rotation angle per pair: pairs of each axis in turn (axis dims 16 / 56 / 56, theta 10000), as EmbedND
void drawRope(const float* ids3, int n, _Float16* P, _Float16* Q) {
    const int dims[3] = {16, 56, 56};
    for (int r = 0; r < n; r++) {
        int k = 0;
        for (int a = 0; a < 3; a++)
            for (int i = 0; i < dims[a] / 2; i++, k++) {
                double ang = ids3[r * 3 + a] * std::pow(10000.0, -2.0 * i / dims[a]);
                P[r * 128 + k] = P[r * 128 + 64 + k] = (_Float16)std::cos(ang);
                Q[r * 128 + k] = Q[r * 128 + 64 + k] = (_Float16)std::sin(ang);
            }
    }
}

}  // namespace

QwenImage::~QwenImage() {
    release();
    if (table_) munmap((void*)table_, tableBytes_);
    if (stable_) munmap((void*)stable_, stableBytes_);
}

void QwenImage::release() {
    for (auto& c : reader_) c.release();
    for (auto& c : part_) c.release();
    dec_.release();
    for (auto& b : bufs_) g_rpc.free(b.second.p);
    bufs_.clear();
    readerLoaded_ = drawingLoaded_ = false;
    nParts_ = 0;
}

Shm& QwenImage::buf(const std::string& key, size_t bytes, std::string* err) {
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

bool QwenImage::bindAll(QnnCtx& c, std::string* err) {
    bool dec = &c == &dec_;
    for (auto& g : c.graphs()) {
        if (want_ && !want_(g.name)) continue;
        for (int io = 0; io < 2; io++) {
            auto& names = io ? g.outNames : g.inNames;
            auto& bytes = io ? g.outBytes : g.inBytes;
            for (size_t i = 0; i < names.size(); i++) {
                std::string key = (dec ? "dec_" : "") + names[i];
                Shm& s = buf(key, bytes[i], err);
                if (!err->empty()) return false;
                if (!c.bind(g, io == 0, i, s, err)) return false;
            }
        }
    }
    return true;
}

// Runs a list of files in order while the next ones load on helper threads (up to `ahead` loading at once):
// loading was most of the reader's and prompt pass's time and a third of the steps'.
bool QwenImage::pipeline(const std::vector<std::string>& paths, int ahead,
                         const std::function<bool(QnnCtx&, std::string*)>& run, double* waitMs, std::string* err) {
    const int n = (int)paths.size();
    std::vector<std::unique_ptr<QnnCtx>> ctxs(n);
    std::vector<std::future<bool>> loads(n);
    std::vector<std::string> errs(n);
    auto start = [&](int k) {
        loads[k] = std::async(std::launch::async, [&, k] {
            ctxs[k].reset(new QnnCtx());
            // in the group anchored on dec.bin: one shared working area instead of each file reserving its own
            if (!ctxs[k]->load(paths[k], &errs[k], groupSpill_, groupSpill_ ? dec_.context() : nullptr, want_)) return false;
            std::lock_guard<std::mutex> lk(bufMutex_);
            return bindAll(*ctxs[k], &errs[k]);
        });
    };
    int next = 0;
    for (; next < std::min(n, ahead); next++) start(next);
    bool ok = true;
    for (int k = 0; k < n; k++) {
        double tw = nowMs();
        if (!loads[k].get()) { *err = errs[k]; ok = false; break; }
        if (waitMs) *waitMs += nowMs() - tw;
        if (next < n) start(next++);
        if (!run(*ctxs[k], err)) { ok = false; break; }
        LOGI("ran %s", paths[k].substr(paths[k].rfind('/') + 1).c_str());
        ctxs[k].reset();
    }
    for (int k = 0; k < n; k++) if (loads[k].valid()) loads[k].wait();
    return ok;
}

bool QwenImage::loadReader(std::string* err) {
    if (readerLoaded_) return true;
    for (int k = 0; k < 4; k++)
        if (!reader_[k].load(dir_ + "/r" + std::to_string(k) + ".bin", err) || !bindAll(reader_[k], err)) return false;
    readerLoaded_ = true;
    return true;
}

void QwenImage::freeReader() {
    for (auto& c : reader_) c.release();
    readerLoaded_ = false;
}

// The drawing model is either part<k>.bin (a few blocks per graph, graphs "prefix" / "s<W>x<H>") or pack<k>.bin
// (several contexts, one block per graph: "prefix_<j>" / "s<W>x<H>_<j>", the faster layout). graphFor() finds the
// graphs to run, in block order, for either.
std::vector<std::pair<QnnCtx*, QnnCtx::Graph*>> QwenImage::graphsFor(const std::string& kind) {
    std::vector<std::pair<QnnCtx*, QnnCtx::Graph*>> out;
    if (!packed_) {
        for (int k = 0; k < nParts_; k++) out.push_back({&part_[k], part_[k].graph(kind)});
        return out;
    }
    for (int j = 0; j < 64; j++) {
        std::string want = kind + "_" + std::to_string(j);
        bool found = false;
        for (int k = 0; k < nParts_ && !found; k++)
            for (auto& g : part_[k].graphs())
                if (g.name == want) { out.push_back({&part_[k], &g}); found = true; break; }
        if (!found) break;
    }
    return out;
}

bool QwenImage::loadDrawing(std::string* err) {
    if (drawingLoaded_) return true;
    if (swap_) {   // only the decoder stays (loaded by anchorGroup()); the per-step files are loaded in turn by generate()
        if (!anchorGroup(err)) return false;
        drawingLoaded_ = true;
        return true;
    }
    nParts_ = 0;
    // spack<k>.bin: per-step graphs only (8 per file, kept loaded); ppack<k>.bin: the prompt pass, loaded per picture.
    split_ = access((draw_ + "/spack0.bin").c_str(), R_OK) == 0;
    packed_ = split_ || access((draw_ + "/pack0.bin").c_str(), R_OK) == 0;
    const std::string stem = split_ ? "/spack" : packed_ ? "/pack" : "/part";
    while (nParts_ < 32 && access((draw_ + stem + std::to_string(nParts_) + ".bin").c_str(), R_OK) == 0) nParts_++;
    if (nParts_ == 0) { *err = "no drawing pieces"; return false; }
    if (split_) {
        // per-step files: one batch call with sequential resource sharing (each file alone reserves ~5.6 GB)
        std::vector<QnnCtx*> cs; std::vector<std::string> ps;
        for (int k = 0; k < nParts_; k++) { cs.push_back(&part_[k]); ps.push_back(draw_ + stem + std::to_string(k) + ".bin"); }
        if (!QnnCtx::loadList(cs, ps, err)) return false;
        for (int k = 0; k < nParts_; k++) if (!bindAll(part_[k], err)) return false;
        if (!dec_.load(dir_ + "/dec.bin", err) || !bindAll(dec_, err)) return false;
        drawingLoaded_ = true;
        return true;
    }
    // the pieces run one after another, so they share one scratch area (each alone reserves ~1 GB at 768)
    uint64_t spill = 0;
    for (int k = 0; k < nParts_; k++) spill = std::max(spill, QnnGraph::spillSize(draw_ + stem + std::to_string(k) + ".bin"));
    for (int k = 0; k < nParts_; k++)
        if (!part_[k].load(draw_ + stem + std::to_string(k) + ".bin", err, spill, k ? part_[0].context() : nullptr) ||
            !bindAll(part_[k], err)) return false;
    LOGI("%d drawing pieces, shared scratch %.0f MB", nParts_, spill / 1e6);
    if (!dec_.load(dir_ + "/dec.bin", err) || !bindAll(dec_, err)) return false;
    drawingLoaded_ = true;
    return true;
}

// The files loaded in turn (reader pieces, sp<k>.bin) join one group of contexts anchored on dec.bin, which stays
// loaded: the group shares one working (spill-fill) area the size of the largest graph's, instead of every file
// reserving one per graph. Off unless QIMG_GROUP=1: the grouped files still each "estimate" ~3 GB and must share one
// memory domain, so the 6th sp file with edit graphs no longer fit (2026-10-08).
bool QwenImage::anchorGroup(std::string* err) {
    if (dec_.context()) return true;
    groupSpill_ = 0;
    const char* g = getenv("QIMG_GROUP");
    if (g && atoi(g) != 0) {
        for (int k = 0; k < 4; k++) groupSpill_ = std::max(groupSpill_, QnnGraph::spillSize(dir_ + "/r" + std::to_string(k) + ".bin"));
        for (int p = 0; access((draw_ + "/sp" + std::to_string(p) + ".bin").c_str(), R_OK) == 0; p++)
            groupSpill_ = std::max(groupSpill_, QnnGraph::spillSize(draw_ + "/sp" + std::to_string(p) + ".bin"));
        LOGI("shared working area %.0f MB", groupSpill_ / 1e6);
    }
    return dec_.load(dir_ + "/dec.bin", err, groupSpill_, nullptr) && bindAll(dec_, err);
}

bool QwenImage::init(const std::string& dir, const std::string& libDir, std::string* err, const std::string& drawDir) {
    if (table_ || stable_) return true;
    dir_ = dir;
    draw_ = !drawDir.empty() && access((drawDir + "/sp0.bin").c_str(), R_OK) == 0 ? drawDir : dir;
    if (const char* a = getenv("QIMG_AHEAD")) ahead_ = std::max(1, atoi(a));
    if (!QnnBackend::get().open(libDir, err)) return false;
    if (!g_rpc.open()) { *err = "libcdsprpc.so not available"; return false; }
    // the 8B reader's word table (Standard download) and/or the 2B reader's (Small download)
    int fd = open((dir + "/qwen_table_q8.raw").c_str(), O_RDONLY);
    if (fd >= 0) {
        tableBytes_ = (size_t)lseek(fd, 0, SEEK_END);
        void* m = mmap(nullptr, tableBytes_, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (m == MAP_FAILED) { *err = "cannot map the word table"; return false; }
        table_ = (const int8_t*)m;
        scale_ = (const float*)(table_ + tableBytes_ / (D + 4) * D);
    }
    int sfd = open((dir + "/qwen2b_table_q8.raw").c_str(), O_RDONLY);
    if (sfd >= 0) {
        stableBytes_ = (size_t)lseek(sfd, 0, SEEK_END);
        void* sm = mmap(nullptr, stableBytes_, PROT_READ, MAP_SHARED, sfd, 0);
        close(sfd);
        if (sm != MAP_FAILED) {
            stable_ = (const int8_t*)sm;
            sscale_ = (const float*)(stable_ + stableBytes_ / (DS + 4) * DS);
        }
    }
    if (!table_ && !stable_) { *err = "no prompt reader (qwen_table_q8.raw or qwen2b_table_q8.raw) in " + dir; return false; }
    return true;
}

bool QwenImage::generate(const std::vector<int32_t>& ids, uint64_t seed, int W, int H, QimgResult* r, std::string* err,
                         const uint8_t* editRgb, const std::function<void(int, int)>& progress) {
    if (progress) progress(0, 4);
    const bool edit = editRgb != nullptr;
    const int TR = edit ? TE : T;                // reader rows
    int n = (int)ids.size();
    if (n > TR) { *err = edit ? "edit prompt longer than 704 words" : "prompt longer than 128 words"; return false; }
    int s0 = -1, seen = 0;                       // the user turn starts at the second <|im_start|>
    for (int i = 0; i < n && s0 < 0; i++) if (ids[i] == 151644 && ++seen == 2) s0 = i;
    if (s0 < 0) { *err = "prompt is not in the chat template"; return false; }
    int img = -1;                                // first of the picture's NV <|image_pad|> rows
    if (edit) {
        for (int i = 0; i < n && img < 0; i++) if (ids[i] == 151655) img = i;
        if (img < s0 || img + NV > n || ids[img + NV - 1] != 151655) { *err = "edit prompt needs 576 <|image_pad|> words"; return false; }
        if (W != 768 || H != 768) { *err = "editing is built for 768x768 only"; return false; }
        if (!table_) { *err = "editing needs the Standard Qwen-Image download (the Small one reads prompts only)"; return false; }
    }
    const int h = H / 16, w = W / 16, N = h * w;
    std::string shape = "s" + std::to_string(W) + "x" + std::to_string(H);
    // only the graphs this run uses are loaded (each file holds the plain and the edit graphs; loading all four
    // graphs of an sp file reserved ~2 GB and doubled the steps' loading time)
    want_ = [edit](const std::string& g) {
        bool perBlock = g.find('_') != std::string::npos;
        if (edit) return !(g == "t128" || g == "s128" || g.rfind("prefix_", 0) == 0 || (g[0] == 's' && perBlock));
        return !(g == "t704" || g.rfind("eprefix_", 0) == 0 || (g[0] == 'e' && perBlock));
    };

    if (access((draw_ + "/sp0.bin").c_str(), R_OK) == 0 && !anchorGroup(err)) return false;

    // ---- the edit picture: vision tower (the reader's eyes) and the tiny encoder (the drawing model's copy)
    double t0 = nowMs();
    if (edit) {
        Shm& px = buf("vpx", (size_t)VP * 768 * 2, err);
        Shm& ei = buf("eimg", (size_t)4 * 768 * 768 * 2, err);
        if (!err->empty()) return false;
        auto* v = (_Float16*)px.p;               // patches in Qwen's order: 2x2 groups row-major, then (channel, y, x)
        for (int p = 0; p < VP; p++) {
            int g = p / 4, by = g / 24, bx = g % 24, my = (p / 2) % 2, mx = p % 2;
            int y0 = (by * 2 + my) * 16, x0 = (bx * 2 + mx) * 16;
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 16; x++)
                        v[(size_t)p * 768 + c * 256 + y * 16 + x] =
                            (_Float16)(editRgb[((size_t)(y0 + y) * 768 + x0 + x) * 3 + c] / 127.5f - 1.f);
        }
        auto* e = (_Float16*)ei.p;
        for (size_t q = 0; q < (size_t)768 * 768; q++) {
            for (int c = 0; c < 3; c++) e[c * 768 * 768 + q] = (_Float16)(editRgb[q * 3 + c] / 255.f);
            e[3 * 768 * 768 + q] = (_Float16)1.f;
        }
        for (const char* f : {"vis", "enc"}) {
            QnnCtx c;
            if (!c.load(dir_ + "/" + f + ".bin", err) || !bindAll(c, err) || !c.exec(c.graphs()[0], err)) return false;
        }
        r->msVision = nowMs() - t0;
    }

    // ---- prompt reader
    t0 = nowMs();
    const char* rsel = getenv("QIMG_READER");
    // the 2B when it is the reader installed (Small download), or when asked for with QIMG_READER=2b
    const bool small = !edit && stable_ && access((dir_ + "/proj.bin").c_str(), R_OK) == 0 &&
                       access((dir_ + "/s1.bin").c_str(), R_OK) == 0 && (!table_ || (rsel && std::string(rsel) == "2b"));
    const std::string rn = small ? "s" : edit ? "re" : "r";   // tensor names: rx<k>/rp/rq/rmask, rex<k>/... + rds<i>, or sx0/sp/sq/smask
    if (small) {
        // the 2B: s0.bin (layers 0-13), s1.bin (14-27), proj.bin (the translator to the 8B's features "ctxall")
        buf("sx0", (size_t)T * DS * 2, err); buf("sp", T * HD * 2, err); buf("sq", T * HD * 2, err); buf("smask", T * T * 2, err);
        if (!err->empty()) return false;
        auto* x = (_Float16*)bufs_["sx0"].p;
        for (int r = 0; r < T; r++) {
            int id = r < n ? ids[r] : 151643;
            const int8_t* q = stable_ + (size_t)id * DS;
            float s = sscale_[id];
            for (int k = 0; k < DS; k++) x[(size_t)r * DS + k] = (_Float16)(q[k] * s);
        }
        auto* P = (_Float16*)bufs_["sp"].p; auto* Q = (_Float16*)bufs_["sq"].p;
        for (int p = 0; p < T; p++)
            for (int i = 0; i < 64; i++) {
                double a = p * std::pow(5e6, -2.0 * i / 128);
                P[p * 128 + i] = P[p * 128 + 64 + i] = (_Float16)std::cos(a);
                Q[p * 128 + i] = Q[p * 128 + 64 + i] = (_Float16)std::sin(a);
            }
        auto* M = (_Float16*)bufs_["smask"].p;
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++) M[i * T + j] = (j <= i && (j < n || j == i)) ? (_Float16)0.f : NEG;
        std::vector<std::string> ps = {dir_ + "/s0.bin", dir_ + "/s1.bin", dir_ + "/proj.bin"};
        if (!pipeline(ps, ahead_, [](QnnCtx& c, std::string* e) { return c.exec(c.graphs()[0], e); }, nullptr, err)) return false;
    } else {
        // the reader's inputs exist before its first piece is loaded (the pieces bind to them by name)
        buf(rn + "x0", (size_t)TR * D * 2, err); buf(rn + "p", (size_t)TR * HD * 2, err); buf(rn + "q", (size_t)TR * HD * 2, err);
        buf(rn + "mask", (size_t)TR * TR * 2, err);
        if (edit) for (int i = 0; i < 3; i++) buf("rds" + std::to_string(i), (size_t)TR * D * 2, err);
        if (!err->empty()) return false;
        auto* x = (_Float16*)bufs_[rn + "x0"].p;
        for (int r = 0; r < TR; r++) {
            if (edit && r >= img && r < img + NV) {
                memcpy(x + (size_t)r * D, (const _Float16*)bufs_["vm"].p + (size_t)(r - img) * D, D * 2);
                continue;
            }
            int id = r < n ? ids[r] : 151643;
            const int8_t* q = table_ + (size_t)id * D;
            float s = scale_[id];
            for (int k = 0; k < D; k++) x[(size_t)r * D + k] = (_Float16)(q[k] * s);
        }
        if (edit)
            for (int i = 0; i < 3; i++) {
                auto* d = (_Float16*)bufs_["rds" + std::to_string(i)].p;
                memset(d, 0, (size_t)TR * D * 2);
                memcpy(d + (size_t)img * D, bufs_["vd" + std::to_string(i)].p, (size_t)NV * D * 2);
            }
        // positions: text counts up; with a picture its rows sit at (img, img + y, img + x) on a 24x24 grid, the text
        // after it continues at img + 24, and the angle pairs alternate axes (Qwen3-VL's interleaved layout)
        auto* P = (_Float16*)bufs_[rn + "p"].p; auto* Q = (_Float16*)bufs_[rn + "q"].p;
        for (int p = 0; p < TR; p++) {
            double pos[3] = {(double)p, (double)p, (double)p};
            if (edit && p >= img && p < img + NV) {
                pos[0] = img; pos[1] = img + (p - img) / 24; pos[2] = img + (p - img) % 24;
            } else if (edit && p >= img + NV) {
                pos[0] = pos[1] = pos[2] = img + 24 + (p - img - NV);
            }
            for (int i = 0; i < 64; i++) {
                int axis = i < 60 ? i % 3 : 0;
                double a = pos[axis] * std::pow(5e6, -2.0 * i / 128);
                P[p * 128 + i] = P[p * 128 + 64 + i] = (_Float16)std::cos(a);
                Q[p * 128 + i] = Q[p * 128 + 64 + i] = (_Float16)std::sin(a);
            }
        }
        auto* M = (_Float16*)bufs_[rn + "mask"].p;
        for (int i = 0; i < TR; i++)
            for (int j = 0; j < TR; j++) M[(size_t)i * TR + j] = (j <= i && (j < n || j == i)) ? (_Float16)0.f : NEG;
        // pieces of ~1.7 GB, each run then released while the next loads
        std::vector<std::string> ps;
        for (int k = 0; k < 4; k++) ps.push_back(dir_ + "/r" + std::to_string(k) + ".bin");
        const std::string gname = edit ? "t704" : "t128";
        auto run = [&gname](QnnCtx& c, std::string* e) {
            QnnCtx::Graph* g = c.graphs().size() == 1 ? &c.graphs()[0] : c.graph(gname);
            if (!g) { *e = "reader graph " + gname + " missing"; return false; }
            return c.exec(*g, e);
        };
        if (!pipeline(ps, ahead_, run, nullptr, err)) return false;
    }
    // prompt rows the drawing model sees: the user turn on, without the picture's rows; the picture goes in at `slot`
    const int nc = n - s0 - (edit ? NV : 0), slot = edit ? img - s0 : 0;
    if (nc > T) { *err = "prompt longer than 128 words"; return false; }
    {
        auto* out = (const _Float16*)bufs_[small ? "ctxall" : rn + "x4"].p;
        Shm& ctx = buf("ctx", (size_t)T * D * 2, err);
        if (!err->empty()) return false;
        auto* c = (_Float16*)ctx.p;
        memset(c, 0, (size_t)T * D * 2);
        if (edit) {
            memcpy(c, out + (size_t)s0 * D, (size_t)slot * D * 2);
            memcpy(c + (size_t)slot * D, out + (size_t)(img + NV) * D, (size_t)(nc - slot) * D * 2);
        } else {
            memcpy(c, out + (size_t)s0 * D, (size_t)nc * D * 2);
        }
    }
    freeReader();
    dumpRaw("ctx", bufs_["ctx"].p, bufs_["ctx"].bytes);
    r->msReader = nowMs() - t0;

    // ---- drawing model. With split files the prompt pass runs before the per-step files are loaded (it needs
    // its own room on the neural unit).
    split_ = access((draw_ + "/spack0.bin").c_str(), R_OK) == 0 || access((draw_ + "/sp0.bin").c_str(), R_OK) == 0;
    swap_ = access((draw_ + "/sp0.bin").c_str(), R_OK) == 0;   // small per-step files, loaded in turn every step
    if (edit && !swap_) { *err = "editing needs the sp<k>.bin drawing files"; return false; }
    // edit: the picture's rows go through every block once with the prompt (t = 0) and their keys/values join the
    // prompt's; the picture sits at axis-0 position `slot` and takes max(h, w) positions
    const int RN = edit ? N : 0, shift = edit ? std::max(h, w) : 0;
    const std::string pre = edit ? "eprefix_" : "prefix_";
    double t1 = nowMs();
    if (!split_ && !loadDrawing(err)) return false;
    r->msLoad = nowMs() - t1;
    double t2 = nowMs();
    {
        buf("pt", T * HD * 2, err); buf("qt", T * HD * 2, err); buf("tmask", T * T * 2, err);
        if (!err->empty()) return false;
        std::vector<float> ids3(T * 3);
        for (int p = 0; p < T; p++) ids3[p * 3] = ids3[p * 3 + 1] = ids3[p * 3 + 2] = (float)p;
        drawRope(ids3.data(), T, (_Float16*)bufs_["pt"].p, (_Float16*)bufs_["qt"].p);
        auto* M = (_Float16*)bufs_["tmask"].p;
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++) M[i * T + j] = (j <= i && (j < nc || j == i)) ? (_Float16)0.f : NEG;
        if (edit) {
            // rotations: ept/eqt for the prompt rows, epr/eqr for the picture's (the graphs keep the two apart)
            buf("ept", T * HD * 2, err); buf("eqt", T * HD * 2, err);
            buf("epr", (size_t)RN * HD * 2, err); buf("eqr", (size_t)RN * HD * 2, err);
            buf("etrb", T * 2, err); buf("ertb", T * 2, err);
            if (!err->empty()) return false;
            std::vector<float> t3(T * 3), r3((size_t)RN * 3);
            for (int p = 0; p < T; p++) t3[p * 3] = t3[p * 3 + 1] = t3[p * 3 + 2] = (float)(p < slot ? p : p + shift);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    float* q = &r3[((size_t)y * w + x) * 3];
                    q[0] = (float)slot; q[1] = (float)(y - (h - h / 2)); q[2] = (float)(x - (w - w / 2));
                }
            drawRope(t3.data(), T, (_Float16*)bufs_["ept"].p, (_Float16*)bufs_["eqt"].p);
            drawRope(r3.data(), RN, (_Float16*)bufs_["epr"].p, (_Float16*)bufs_["eqr"].p);
            auto* tb = (_Float16*)bufs_["etrb"].p; auto* rb = (_Float16*)bufs_["ertb"].p;
            for (int j = 0; j < T; j++) {
                tb[j] = (j >= slot && j < nc) ? (_Float16)0.f : NEG;   // prompt rows after the slot see the picture
                rb[j] = j < slot ? (_Float16)0.f : NEG;                 // the picture sees the prompt before its slot
            }
        }
        if (split_) {
            // the prompt pass: its own ppack<k>.bin files if present, else the sp<k>.bin files, which hold each
            // block's prompt-pass graph next to its step graph (one copy of the weights). One file at a time.
            int j = 0;
            std::vector<std::string> ps;
            const std::string pstem = access((draw_ + "/ppack0.bin").c_str(), R_OK) == 0 ? "/ppack" : "/sp";
            for (int p = 0; access((draw_ + pstem + std::to_string(p) + ".bin").c_str(), R_OK) == 0; p++)
                ps.push_back(draw_ + pstem + std::to_string(p) + ".bin");
            auto runAll = [&j, &pre](QnnCtx& c, std::string* e) {
                for (;; j++) {
                    QnnCtx::Graph* g = nullptr;
                    for (auto& gg : c.graphs()) if (gg.name == pre + std::to_string(j)) g = &gg;
                    if (!g) return true;
                    if (!c.exec(*g, e)) return false;
                }
            };
            if (!pipeline(ps, ahead_, runAll, nullptr, err)) return false;
            if (j == 0) { *err = "no prefix graphs"; return false; }
        } else {
            auto gs = graphsFor("prefix");
            if (gs.empty()) { *err = "no prefix graphs"; return false; }
            for (auto& cg : gs)
                if (!cg.second || !cg.first->exec(*cg.second, err)) { if (err->empty()) *err = "no prefix graph"; return false; }
        }
    }
    r->msPrefix = nowMs() - t2;
    for (const char* n : {"k0", "v0", "k31", "v31", "xt32"}) if (bufs_.count(n)) dumpRaw(n, bufs_[n].p, bufs_[n].bytes);
    if (split_) {
        double tl = nowMs();
        if (!loadDrawing(err)) return false;
        r->msLoad = nowMs() - tl;
    }

    double t3 = nowMs();
    // the per-step inputs exist before any per-step file is loaded (with small files they load mid-step)
    const std::string kbn = edit ? "ekb" : "kb";
    buf("pi", (size_t)N * HD * 2, err); buf("qi", (size_t)N * HD * 2, err); buf(kbn, (size_t)(T + RN) * 2, err);
    buf("lat", (size_t)N * C * 2, err); buf("ts", 256 * 2, err);
    if (!err->empty()) return false;
    {
        std::vector<float> ids3((size_t)N * 3);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                float* p = &ids3[((size_t)y * w + x) * 3];
                p[0] = (float)(nc + shift); p[1] = (float)(y - (h - h / 2)); p[2] = (float)(x - (w - w / 2));
            }
        drawRope(ids3.data(), N, (_Float16*)bufs_["pi"].p, (_Float16*)bufs_["qi"].p);
        auto* kb = (_Float16*)bufs_[kbn].p;
        for (int j = 0; j < T + RN; j++) kb[j] = (j < nc || j >= T) ? (_Float16)0.f : NEG;
    }
    // noise and the turbo schedule (raw nodes 1, .75, .5, .25 shifted for the picture size, then 0)
    std::vector<float> z((size_t)N * C);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : z) v = nd(rng);
    double mu = 0.5 + 0.4 * (N - 256) / (8192.0 - 256.0);
    const double nodes[4] = {1.0, 0.75, 0.5, 0.25};
    double sig[5];
    for (int i = 0; i < 4; i++) sig[i] = std::exp(mu) / (std::exp(mu) + (1.0 / nodes[i] - 1.0));
    sig[4] = 0;
    for (int i = 0; i < 4; i++) {
        if (progress) progress(i + 1, 4);
        auto* lat = (_Float16*)bufs_["lat"].p;
        for (size_t k = 0; k < z.size(); k++) lat[k] = (_Float16)z[k];
        auto* ts = (_Float16*)bufs_["ts"].p;
        for (int k = 0; k < 128; k++) {
            double a = sig[i] * 1000.0 * std::exp(-std::log(10000.0) * k / 128.0);
            ts[k] = (_Float16)std::cos(a); ts[128 + k] = (_Float16)std::sin(a);
        }
        if (swap_) {
            int j = 0;
            std::vector<std::string> ps;
            for (int p = 0; access((draw_ + "/sp" + std::to_string(p) + ".bin").c_str(), R_OK) == 0; p++)
                ps.push_back(draw_ + "/sp" + std::to_string(p) + ".bin");
            const std::string sg = (edit ? "e" + shape.substr(1) : shape) + "_";
            auto runAll = [&j, &sg](QnnCtx& c, std::string* e) {
                for (;; j++) {
                    QnnCtx::Graph* g = nullptr;
                    for (auto& gg : c.graphs()) if (gg.name == sg + std::to_string(j)) g = &gg;
                    if (!g) return true;
                    if (!c.exec(*g, e)) return false;
                }
            };
            if (!pipeline(ps, ahead_, runAll, &r->msSwapLoad, err)) return false;
            if (j == 0) { *err = "no graphs " + sg + "<k>"; return false; }
        } else {
            auto gs = graphsFor(shape);
            if (gs.empty()) { *err = "no graphs for " + shape; return false; }
            for (auto& cg : gs) {
                if (!cg.second) { *err = "no graph " + shape; return false; }
                if (!cg.first->exec(*cg.second, err)) return false;
            }
        }
        LOGI("step %d of 4 done", i + 1);
        if (i == 0) for (const char* n : {"lat", "ts", "pi", "kb", "x1", "mods", "x16", "x17", "x18", "x20", "x24", "x28", "x31", "vel"}) if (bufs_.count(n)) dumpRaw(n, bufs_[n].p, bufs_[n].bytes);
        auto* vel = (const _Float16*)bufs_["vel"].p;
        for (size_t k = 0; k < z.size(); k++) z[k] += (float)(sig[i + 1] - sig[i]) * (float)vel[k];
    }
    r->msSteps = nowMs() - t3;

    double t4 = nowMs();
    {
        auto* lat = (_Float16*)bufs_["dec_lat"].p;            // (64, h, w) from rows (N, 64)
        for (int p = 0; p < N; p++)
            for (int c = 0; c < C; c++) lat[(size_t)c * N + p] = (_Float16)z[(size_t)p * C + c];
        QnnCtx::Graph* g = dec_.graph(shape);
        LOGI("decoding %s", shape.c_str());
        if (!g || !dec_.exec(*g, err)) { if (err->empty()) *err = "no decoder for " + shape; return false; }
        LOGI("decoded");
        auto* rgb = (const _Float16*)bufs_["dec_rgb"].p;
        const size_t NP = (size_t)W * H;
        r->argb.resize(NP);
        auto u8 = [](float v) { return (uint32_t)std::lround(std::min(1.f, std::max(0.f, v)) * 255.f); };
        for (size_t p = 0; p < NP; p++)
            r->argb[p] = 0xFF000000u | (u8((float)rgb[p]) << 16) | (u8((float)rgb[NP + p]) << 8) | u8((float)rgb[2 * NP + p]);
    }
    r->msDecode = nowMs() - t4;
    return true;
}
