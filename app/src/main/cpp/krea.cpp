#include "krea.h"

#include <android/log.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <random>
#include <sstream>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "anima", __VA_ARGS__)

namespace {

constexpr int T = 128;             // reader slots: the template's system turn and "<|im_start|>user\n" first
constexpr int DROP = 34;           // those rows, left out of what the drawing model reads
constexpr int TS = 96;             // prompt slots of the drawing model (rows DROP.., then padding)
constexpr int QH = 2560, NTAP = 12, D = 6144, NB = 28, NMOD = NB + 1;
constexpr int32_t PAD = 151643;
constexpr size_t ROWS = 151936;
constexpr float NEG = -30000.f;

// Debugging: with KREA_DUMP=<dir> set, stage outputs are written there as float32 files.
void dump(const char* name, const std::vector<float>& v) {
    const char* d = getenv("KREA_DUMP");
    if (!d) return;
    std::ofstream(std::string(d) + "/" + name + ".f32", std::ios::binary).write(reinterpret_cast<const char*>(v.data()), v.size() * 4);
}

double nowMs() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const void* mapFile(const std::string& path, size_t* bytes) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    off_t n = lseek(fd, 0, SEEK_END);
    void* p = mmap(nullptr, (size_t)n, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return nullptr;
    *bytes = (size_t)n;
    return p;
}

bool runNamed(QnnGraph& g, const std::map<std::string, const float*>& in, const std::map<std::string, float*>& out,
              std::string* err) {
    std::vector<const float*> ip;
    std::vector<float*> op;
    for (auto& i : g.inputs()) {
        auto it = in.find(i.name);
        if (it == in.end()) { *err = "piece needs unknown input " + i.name; return false; }
        ip.push_back(it->second);
    }
    for (auto& o : g.outputs()) {
        auto it = out.find(o.name);
        if (it == out.end()) { *err = "piece has unknown output " + o.name; return false; }
        op.push_back(it->second);
    }
    return g.run(ip, op, err);
}

inline uint32_t argb(float r, float g, float b) {
    auto q = [](float v) { return (uint32_t)std::lround(std::fmin(std::fmax(v, 0.f), 1.f) * 255.f); };
    return 0xFF000000u | (q(r) << 16) | (q(g) << 8) | q(b);
}

// Rope tables in the half-split layout (out = x*P + swap(x)*Q, swap carrying the minus), 64 angles per position.
// Reader: Qwen3-VL's positions are plain for text (theta 5,000,000).
void ropeText(std::vector<float>* P, std::vector<float>* Q) {
    P->resize(T * 128); Q->resize(T * 128);
    for (int p = 0; p < T; p++)
        for (int i = 0; i < 64; i++) {
            double a = p / std::pow(5e6, 2.0 * i / 128.0);
            (*P)[p * 128 + i] = (*P)[p * 128 + 64 + i] = (float)std::cos(a);
            (*Q)[p * 128 + i] = (*Q)[p * 128 + 64 + i] = (float)std::sin(a);
        }
}

// Drawing: prompt rows at position 0, picture rows (0, y, x); axes of 16 / 24 / 24 angles, theta 1000.
void ropeDraw(int h, int w, std::vector<float>* P, std::vector<float>* Q) {
    const int L = TS + h * w;
    P->assign((size_t)L * 128, 1.f); Q->assign((size_t)L * 128, 0.f);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const size_t row = (size_t)(TS + y * w + x) * 128;
            for (int i = 0; i < 64; i++) {
                double ang = 0;
                if (i >= 16 && i < 40) ang = y * std::pow(1000.0, -2.0 * (i - 16) / 48.0);
                else if (i >= 40) ang = x * std::pow(1000.0, -2.0 * (i - 40) / 48.0);
                (*P)[row + i] = (*P)[row + 64 + i] = (float)std::cos(ang);
                (*Q)[row + i] = (*Q)[row + 64 + i] = (float)std::sin(ang);
            }
        }
}

}  // namespace

bool KreaPipeline::init(const std::string& modelDir, const std::string& libDir, std::string* err) {
    dir_ = modelDir;
    if (!QnnBackend::get().open(libDir, err)) return false;
    release();
    if (table_) munmap(const_cast<int8_t*>(table_), tableBytes_);
    table_ = static_cast<const int8_t*>(mapFile(dir_ + "/krea_qwen_q8.raw", &tableBytes_));
    if (!table_ || tableBytes_ != ROWS * QH + ROWS * 4) { *err = "krea_qwen_q8.raw missing or wrong size"; return false; }
    scale_ = reinterpret_cast<const float*>(table_ + ROWS * QH);
    if (mods_) munmap(const_cast<float*>(mods_), modsBytes_);
    mods_ = static_cast<const float*>(mapFile(dir_ + "/mods.f32", &modsBytes_));
    sig_.clear();
    { std::ifstream f(dir_ + "/sigmas.txt"); float v; while (f >> v) sig_.push_back(v); }
    const size_t steps = sig_.size() ? sig_.size() - 1 : 0;
    if (steps < 1 || !mods_ || modsBytes_ != steps * NMOD * 6 * D * 4) { *err = "mods.f32 / sigmas.txt missing or not matching"; return false; }
    return true;
}

void KreaPipeline::release() { dec_.release(); }

bool KreaPipeline::runText(const KreaRequest& req, std::vector<float>* txt, std::vector<float>* kbText, std::string* err) {
    const int n = (int)req.ids.size();
    if (n <= DROP || n > T || n - DROP > TS) { *err = "prompt too long (" + std::to_string(n) + " > " + std::to_string(T) + " pieces)"; return false; }
    std::vector<float> x((size_t)T * QH), P, Q;
    for (int r = 0; r < T; r++) {
        int32_t id = r < n ? req.ids[r] : PAD;
        if (id < 0 || (size_t)id >= ROWS) { *err = "bad Qwen id"; return false; }
        const int8_t* row = table_ + (size_t)id * QH;
        for (int k = 0; k < QH; k++) x[(size_t)r * QH + k] = row[k] * scale_[id];
    }
    ropeText(&P, &Q);
    { std::vector<float> idf(req.ids.begin(), req.ids.end()); dump("ids", idf); }
    const int kept = n - DROP;
    std::vector<float> taps((size_t)TS * NTAP * QH, 0.f), t[4];
    for (int k = 0; k < 3; k++) {
        QnnGraph g;
        if (!g.load(dir_ + "/text" + std::to_string(k) + ".bin", err)) return false;
        for (auto& v : t) v.assign((size_t)T * QH, 0.f);
        std::vector<float> xo((size_t)T * QH);
        std::map<std::string, float*> out{{"t0", t[0].data()}, {"t1", t[1].data()}, {"t2", t[2].data()}, {"t3", t[3].data()}};
        if (k < 2) out["x_out"] = xo.data();
        if (!runNamed(g, {{"x", x.data()}, {"P", P.data()}, {"Q", Q.data()}}, out, err)) return false;
        for (int j = 0; j < 4; j++)   // tap 4k + j of the kept rows, side by side per row
            for (int r = 0; r < kept; r++)
                std::copy(t[j].begin() + (size_t)(DROP + r) * QH, t[j].begin() + (size_t)(DROP + r + 1) * QH,
                          taps.begin() + ((size_t)r * NTAP + 4 * k + j) * QH);
        if (k < 2) x.swap(xo);
    }
    dump("taps", taps);
    kbText->assign(TS, 0.f);
    for (int r = kept; r < TS; r++) (*kbText)[r] = NEG;
    QnnGraph f;
    if (!f.load(dir_ + "/fuse.bin", err)) return false;
    txt->assign((size_t)TS * D, 0.f);
    if (!runNamed(f, {{"taps", taps.data()}, {"kb", kbText->data()}}, {{"txt", txt->data()}}, err)) return false;
    dump("txt", *txt);
    return true;
}

KreaResult KreaPipeline::generate(const KreaRequest& req, const std::function<void(int, int)>& progress) {
    KreaResult res;
    std::string& err = res.error;
    const int W = req.width, H = req.height, hp = H / 16, wp = W / 16, Ni = hp * wp, L = TS + Ni;
    const int lh = H / 8, lw = W / 8;
    if (W % 16 || H % 16) { err = "size must be a multiple of 16"; return res; }
    const std::string key = "s" + std::to_string(W) + "x" + std::to_string(H);

    double t0 = nowMs();
    std::vector<float> txt, kbT;
    if (!runText(req, &txt, &kbT, &err)) return res;
    double t1 = nowMs();
    res.msText = t1 - t0;

    std::vector<float> P, Q, kb(L, 0.f);
    ropeDraw(hp, wp, &P, &Q);
    std::copy(kbT.begin(), kbT.end(), kb.begin());
    std::vector<float> lat((size_t)16 * lh * lw), img((size_t)Ni * 64), x((size_t)L * D), x2((size_t)L * D), v((size_t)Ni * 64);
    std::mt19937_64 rng(req.seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& e : lat) e = nd(rng);                 // sigma starts at 1: the latent is the noise itself
    // patch j = (py, px), feature c * 4 + dy * 2 + dx  <->  latent (c, 2 py + dy, 2 px + dx)
    auto patchIdx = [&](int j, int f) {
        const int py = j / wp, px = j % wp, c = f / 4, dy = (f / 2) % 2, dx = f % 2;
        return ((size_t)c * lh + 2 * py + dy) * lw + 2 * px + dx;
    };
    // the drawing blocks, each loaded on a helper thread while the one before it runs
    const int steps = (int)sig_.size() - 1;
    std::vector<std::unique_ptr<QnnGraph>> blk(NB);
    std::vector<std::future<bool>> load(NB);
    std::vector<std::string> lerr(NB);
    auto start = [&](int k) {
        load[k] = std::async(std::launch::async, [&, k] {
            blk[k].reset(new QnnGraph());
            return blk[k]->load(dir_ + "/b" + std::to_string(k) + ".bin", &lerr[k], false, nullptr, 0, key + "_" + std::to_string(k));
        });
    };
    for (int s = 0; s < steps; s++) {
        const float sg = sig_[s], sn = sig_[s + 1];
        for (int j = 0; j < Ni; j++)
            for (int f = 0; f < 64; f++) img[(size_t)j * 64 + f] = lat[patchIdx(j, f)];
        if (s == 0) dump("img0", img);
        start(0);
        for (int i = 0; i < NB; i++) {
            double tw = nowMs();
            bool ok = load[i].get();
            res.msWait += nowMs() - tw;
            if (!ok) { err = "block " + std::to_string(i) + ": " + lerr[i]; return res; }
            if (i + 1 < NB) start(i + 1);
            const float* m = mods_ + ((size_t)s * NMOD + i) * 6 * D;
            const float* mf = mods_ + ((size_t)s * NMOD + NB) * 6 * D;
            bool ran;
            if (i == 0)
                ran = runNamed(*blk[i], {{"img", img.data()}, {"txt", txt.data()}, {"m", m}, {"P", P.data()}, {"Q", Q.data()}, {"kb", kb.data()}},
                               {{"x_out", x.data()}}, &err);
            else if (i < NB - 1)
                ran = runNamed(*blk[i], {{"x", x.data()}, {"m", m}, {"P", P.data()}, {"Q", Q.data()}, {"kb", kb.data()}},
                               {{"x_out", x2.data()}}, &err);
            else
                ran = runNamed(*blk[i], {{"x", x.data()}, {"m", m}, {"P", P.data()}, {"Q", Q.data()}, {"kb", kb.data()}, {"mf", mf}},
                               {{"v", v.data()}}, &err);
            if (!ran) { for (auto& l : load) if (l.valid()) l.wait(); return res; }
            if (i > 0 && i < NB - 1) x.swap(x2);
            if (s == 0 && i == 0) dump("x0", x);
            blk[i].reset();
        }
        if (s == 0) dump("v0", v);
        for (int j = 0; j < Ni; j++)
            for (int f = 0; f < 64; f++) lat[patchIdx(j, f)] += (sn - sg) * v[(size_t)j * 64 + f];   // Euler, flow velocity
        progress(s + 1, steps);
    }
    dump("latN", lat);
    double t2 = nowMs();
    res.msDraw = t2 - t1;

    if (!dec_.loaded() && !dec_.load(dir_ + "/dec.bin", &err, false, nullptr, 0, key)) return res;
    const int N = W * H;
    std::vector<float> rgb(3 * (size_t)N);
    if (!runNamed(dec_, {{"lat", lat.data()}}, {{"rgb", rgb.data()}}, &err)) return res;
    res.argb.resize(N);
    for (int p = 0; p < N; p++) res.argb[p] = argb(rgb[p], rgb[N + p], rgb[2 * (size_t)N + p]);
    res.msDecode = nowMs() - t2;
    return res;
}
