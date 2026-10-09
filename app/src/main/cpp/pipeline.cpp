#include "pipeline.h"

#include <android/log.h>
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "anima", __VA_ARGS__)

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <algorithm>
#include <random>

namespace {

constexpr int NQ = 256, NT = 256, H = 1024;  // prompt reader slots
constexpr int D = 2048;                      // drawing model width
constexpr int CDIM = 32;                     // guide embedding width (control add-on)
constexpr int32_t QWEN_PAD = 151643;
constexpr float NEG = -30000.f;

double nowMs() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const uint16_t* mapFile(const std::string& path, size_t* bytes) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    off_t n = lseek(fd, 0, SEEK_END);
    void* p = mmap(nullptr, (size_t)n, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return nullptr;
    *bytes = (size_t)n;
    return static_cast<const uint16_t*>(p);
}

inline float h2f(uint16_t h) {
    _Float16 v;
    memcpy(&v, &h, 2);
    return (float)v;
}

// Feed inputs/outputs by name: the graph's own order is whatever the converter chose.
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

// Timestep embedding as in comfy/ldm/cosmos/predict2.py Timesteps: [cos | sin], computed in float32.
void sinusoid(float sigma, float* out) {
    const int half = D / 2;
    for (int i = 0; i < half; i++) {
        float a = sigma * std::exp(-std::log(10000.f) * (float)i / (float)half);
        out[i] = std::cos(a);
        out[half + i] = std::sin(a);
    }
}

// (16,2hp,2wp) latent -> (hp*wp,68): 'b c (h m) (w n) -> b (h w) (c m n)' with a zero padding-mask channel.
void patchify(const float* lat, float* x, int hp, int wp) {
    const int lh = 2 * hp, lw = 2 * wp;
    for (int h = 0; h < hp; h++)
        for (int w = 0; w < wp; w++) {
            float* row = x + (h * wp + w) * 68;
            for (int c = 0; c < 17; c++)
                for (int m = 0; m < 2; m++)
                    for (int n = 0; n < 2; n++)
                        row[c * 4 + m * 2 + n] = c < 16 ? lat[(c * lh + 2 * h + m) * lw + 2 * w + n] : 0.f;
        }
}

// (hp*wp,64) -> (16,2hp,2wp): 'B (H W) (p1 p2 C) -> B C (H p1) (W p2)'.
void unpatchify(const float* o, float* lat, int hp, int wp) {
    const int lh = 2 * hp, lw = 2 * wp;
    for (int h = 0; h < hp; h++)
        for (int w = 0; w < wp; w++) {
            const float* row = o + (h * wp + w) * 64;
            for (int p1 = 0; p1 < 2; p1++)
                for (int p2 = 0; p2 < 2; p2++)
                    for (int c = 0; c < 16; c++)
                        lat[(c * lh + 2 * h + p1) * lw + 2 * w + p2] = row[p1 * 32 + p2 * 16 + c];
        }
}

// ComfyUI's ModelSamplingDiscreteFlow.percent_to_sigma (shift 3): the guide is used while sigma >= this.
float percentToSigma(float p) {
    if (p <= 0.f) return 1.f;
    if (p >= 1.f) return 0.f;
    float t = 1.f - p;
    return 3.f * t / (1.f + 2.f * t);
}

inline uint32_t argb(float r, float g, float b) {
    auto q = [](float v) { return (uint32_t)std::lround(std::fmin(std::fmax(v, 0.f), 1.f) * 255.f); };
    return 0xFF000000u | (q(r) << 16) | (q(g) << 8) | q(b);
}

}  // namespace

// ComfyUI's ModelSamplingDiscreteFlow (shift 3, multiplier 1) table + the "simple" scheduler.
std::vector<float> Pipeline::sigmas(int steps) {
    const double shift = 3.0;
    std::vector<float> tab(1000);
    for (int i = 0; i < 1000; i++) {
        double t = (i + 1) / 1000.0;
        tab[i] = (float)(shift * t / (1.0 + (shift - 1.0) * t));
    }
    std::vector<float> s;
    double ss = 1000.0 / steps;
    for (int x = 0; x < steps; x++) s.push_back(tab[999 - (int)(x * ss)]);
    s.push_back(0.f);
    return s;
}

bool Pipeline::init(const std::string& modelDir, const std::string& libDir, std::string* err) {
    dir_ = modelDir;
    if (!QnnBackend::get().open(libDir, err)) return false;
    // Called again after a model download: whatever was loaded or mapped belongs to the old files
    // (the Low set has its own, rotated Qwen table, so a kept mapping garbled every prompt after a switch).
    release();
    if (qwenTable_) munmap(const_cast<uint16_t*>(qwenTable_), qwenBytes_);
    if (qwenQ8_) munmap(const_cast<int8_t*>(qwenQ8_), qwenBytes_);
    qwenTable_ = nullptr; qwenQ8_ = nullptr; qwenScale_ = nullptr;
    if (t5Table_) munmap(const_cast<uint16_t*>(t5Table_), t5Bytes_);
    constexpr size_t QROWS = 151936;
    if (access((dir_ + "/qwen_embed_q8.raw").c_str(), R_OK) == 0) {
        qwenQ8_ = reinterpret_cast<const int8_t*>(mapFile(dir_ + "/qwen_embed_q8.raw", &qwenBytes_));
        if (!qwenQ8_ || qwenBytes_ != QROWS * H + QROWS * 4) { *err = "qwen_embed_q8.raw missing or wrong size"; return false; }
        qwenScale_ = reinterpret_cast<const float*>(qwenQ8_ + QROWS * H);
    } else {
        qwenTable_ = mapFile(dir_ + "/qwen_embed_f16.raw", &qwenBytes_);
    }
    t5Table_ = mapFile(dir_ + "/t5_embed_f16.raw", &t5Bytes_);
    if (!qwenQ8_ && (!qwenTable_ || qwenBytes_ != QROWS * H * 2)) { *err = "qwen_embed_f16.raw missing or wrong size"; return false; }
    if (!t5Table_ || t5Bytes_ != (size_t)32128 * H * 2) { *err = "t5_embed_f16.raw missing or wrong size"; return false; }
    // Sizes: the graphs of the drawing pieces ("s<W>x<H>"); an older single-size download has one graph.
    sizes_.clear();
    for (const std::string& n : QnnGraph::graphNames(dir_ + "/part0.bin"))
        if (n.size() > 1 && n[0] == 's' && n.find('x') != std::string::npos) sizes_.push_back(n.substr(1));
    if (sizes_.empty()) sizes_.push_back("512x512");
    guideFile_ = access((dir_ + "/guide.bin").c_str(), R_OK) == 0;
    w_ = h_ = 0;
    return true;
}

// Picks each file's graph for this size and sizes the shared scratch buffer for it. The decoder and the four
// drawing pieces run one after another, so they share one scratch buffer sized for the largest of them
// (saves ~0.2 GB over one buffer each); bigger pictures need a bigger one, so changing size reloads them all.
bool Pipeline::useSize(int w, int h, std::string* err) {
    if (w == w_ && h == h_) return true;
    std::string key = std::to_string(w) + "x" + std::to_string(h);
    if (std::find(sizes_.begin(), sizes_.end(), key) == sizes_.end()) { *err = "this model has no " + key + " size"; return false; }
    auto pick = [&](const std::string& file, std::string* g) {
        std::vector<std::string> names = QnnGraph::graphNames(dir_ + "/" + file);
        if (std::find(names.begin(), names.end(), "s" + key) != names.end()) { *g = "s" + key; return true; }
        if (names.size() == 1 && key == "512x512") { g->clear(); return true; }  // older single-size file
        *err = file + " has no " + key + " size";
        return false;
    };
    release();
    if (!pick("part0.bin", &partG_) || !pick("dec.bin", &decG_)) return false;
    if (guideFile_ && !pick("guide.bin", &guideG_)) return false;
    // Sized for the largest size in the files, not just this one: with a smaller buffer the decoder's context
    // would not load at 512 (contextCreateFromBinary failed) although only its 512 graph is enabled.
    groupSpill_ = QnnGraph::spillSize(dir_ + "/dec.bin");
    for (int k = 0; k < 4; k++)
        groupSpill_ = std::max(groupSpill_, QnnGraph::spillSize(dir_ + "/part" + std::to_string(k) + ".bin"));
    w_ = w; h_ = h;
    return true;
}

// The guide picture -> guide embedding (L x CDIM), once per picture. Grey copy as in the PC tests (PIL's
// "L" weights), values 0..1, same in all three channels.
bool Pipeline::runGuide(const GenRequest& req, std::vector<float>* cemb, std::string* err) {
    const int L = (w_ / 16) * (h_ / 16), N = w_ * h_;
    cemb->assign((size_t)L * CDIM, 0.f);
    if (req.guide.empty()) return true;
    if (!guideFile_) { *err = "this model cannot follow a guide picture (guide.bin missing)"; return false; }
    if (req.guide.size() != (size_t)N) { *err = "guide picture has the wrong size"; return false; }
    std::vector<float> img(3 * (size_t)N);
    for (int p = 0; p < N; p++) {
        uint32_t c = req.guide[p];
        float g = (299.f * ((c >> 16) & 255) + 587.f * ((c >> 8) & 255) + 114.f * (c & 255)) / 1000.f;
        g = std::floor(g + 0.5f) / 255.f;
        img[p] = img[N + p] = img[2 * N + p] = g;
    }
    QnnGraph gr;
    if (!gr.load(dir_ + "/guide.bin", err, false, nullptr, 0, guideG_)) return false;
    if (gr.outputs().size() != 1 || gr.outputs()[0].count != cemb->size()) { *err = "guide reader has an unexpected shape"; return false; }
    return runNamed(gr, {{"img", img.data()}}, {{"cemb", cemb->data()}}, err);
}

bool Pipeline::runText(const GenRequest& req, std::vector<float>* ctx, std::string* err) {
    int tq = (int)req.qwenIds.size(), tt = (int)req.t5Ids.size();
    if (tq < 1 || tq > NQ) { *err = "prompt too long for the Qwen reader (" + std::to_string(tq) + " > 256 pieces)"; return false; }
    if (tt < 1 || tt > NT - 1) { *err = "prompt too long (" + std::to_string(tt) + " > 255 pieces)"; return false; }
    std::vector<float> qe((size_t)NQ * H), te((size_t)NT * H, 0.f), sb(NQ), tb(NT), keep(NT);
    for (int r = 0; r < NQ; r++) {
        int32_t id = r < tq ? req.qwenIds[r] : QWEN_PAD;
        if (id < 0 || id >= 151936) { *err = "bad Qwen id"; return false; }
        if (qwenQ8_) {
            const int8_t* row = qwenQ8_ + (size_t)id * H;
            const float sc = qwenScale_[id];
            for (int k = 0; k < H; k++) qe[(size_t)r * H + k] = row[k] * sc;
        } else {
            const uint16_t* row = qwenTable_ + (size_t)id * H;
            for (int k = 0; k < H; k++) qe[(size_t)r * H + k] = h2f(row[k]);
        }
        sb[r] = r < tq ? 0.f : NEG;
    }
    for (int r = 0; r < NT; r++) {
        if (r < tt) {
            int32_t id = req.t5Ids[r];
            if (id < 0 || id >= 32128) { *err = "bad T5 id"; return false; }
            const uint16_t* row = t5Table_ + (size_t)id * H;
            for (int k = 0; k < H; k++) te[(size_t)r * H + k] = h2f(row[k]);
        }
        tb[r] = r < tt ? 0.f : NEG;
        // ComfyUI multiplies the adapter's output rows by the emphasis weights; padded rows are zero.
        keep[r] = r < tt ? (req.t5Weights.size() == (size_t)tt ? req.t5Weights[r] : 1.f) : 0.f;
    }
    if (!text_.loaded() && !text_.load(dir_ + "/text.bin", err)) return false;
    ctx->assign((size_t)NT * H, 0.f);
    bool ok = runNamed(text_, {{"qwen_emb", qe.data()}, {"src_bias", sb.data()}, {"t5_emb", te.data()},
                               {"tgt_bias", tb.data()}, {"keep", keep.data()}},
                       {{"ctx", ctx->data()}}, err);
    if (mode_ != 0) text_.release();  // never resident together with the drawing pieces unless "fastest"
    return ok;
}

// The decoder is loaded first and stays while drawing: it anchors the shared scratch group, so drawing
// pieces can come and go (low memory modes) without the group disappearing.
bool Pipeline::loadPart(int k, std::string* err) {
    if (part_[k].loaded()) return true;
    bool g = groupSpill_ > 0 && dec_.loaded();
    return part_[k].load(dir_ + "/part" + std::to_string(k) + ".bin", err, g, g ? dec_.context() : nullptr, 0, partG_);
}

bool Pipeline::loadDraw(std::string* err) {
    if (!dec_.loaded() && !dec_.load(dir_ + "/dec.bin", err, groupSpill_ > 0, nullptr, groupSpill_, decG_)) {
        if (groupSpill_ == 0) return false;
        // Safety net: if the shared scratch group cannot be set up, every piece gets its own scratch
        // (more memory, but the picture still comes out).
        LOGW("shared scratch failed (%s); loading without it", err->c_str());
        groupSpill_ = 0;
        err->clear();
        if (!dec_.load(dir_ + "/dec.bin", err, false, nullptr, 0, decG_)) return false;
    }
    for (int k = 0; k < resident(); k++)
        if (!loadPart(k, err)) return false;
    return true;
}

void Pipeline::release() {
    text_.release();
    for (auto& p : part_) p.release();
    dec_.release();  // last: it anchors the scratch group
}

GenResult Pipeline::generate(const GenRequest& req, const std::function<void(int, int)>& progress) {
    GenResult res;
    std::string& err = res.error;
    if (!useSize(req.width, req.height, &err)) return res;
    const int W = w_, Hh = h_, hp = Hh / 16, wp = W / 16, L = hp * wp;
    // Strict stages for the 2-3 GB memory goal: drawing pieces out, prompt reader in and out, drawing back in.
    if (mode_ != 0) {
        for (auto& p : part_) p.release();
        dec_.release();
    }

    double t0 = nowMs();
    std::vector<float> ctx, cemb;
    if (!runText(req, &ctx, &err)) return res;
    if (!runGuide(req, &cemb, &err)) return res;
    double t1 = nowMs();
    res.msText = t1 - t0;
    if (!loadDraw(&err)) return res;
    double t2 = nowMs();
    res.msLoad = t2 - t1;

    const size_t NL = (size_t)16 * (Hh / 8) * (W / 8);
    std::vector<float> lat(NL);
    std::vector<float> sig = sigmas(req.steps);
    if (req.noise.size() == NL) {
        for (size_t i = 0; i < NL; i++) lat[i] = req.noise[i] * sig[0];
    } else {
        std::mt19937_64 rng(req.seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (size_t i = 0; i < NL; i++) lat[i] = nd(rng) * sig[0];
    }
    std::vector<float> x((size_t)L * 68), emb(D), h((size_t)L * D), h2((size_t)L * D), embN(D), lora(3 * D),
        out((size_t)L * 64), v(NL), st(1);
    const float cut = percentToSigma(req.guideEnd);
    for (int i = 0; i < req.steps; i++) {
        float s = sig[i], sn = sig[i + 1];
        sinusoid(s, emb.data());
        patchify(lat.data(), x.data(), hp, wp);
        // Strength 0 gives exactly the plain picture, so steps outside the guide's part simply pass 0.
        st[0] = !req.guide.empty() && s >= cut - 1e-6f ? req.guideStrength : 0.f;
        // In the low memory modes the pieces past resident() are loaded just for their turn and freed again.
        auto done = [&](int k) { if (k >= resident()) part_[k].release(); };
        if (!loadPart(0, &err)) return res;
        if (!runNamed(part_[0], {{"x", x.data()}, {"emb", emb.data()}, {"ctx", ctx.data()}, {"cemb", cemb.data()}, {"st", st.data()}},
                      {{"x_out", h.data()}, {"emb_out", embN.data()}, {"lora_out", lora.data()}}, &err)) return res;
        done(0);
        for (int k = 1; k <= 2; k++) {
            if (!loadPart(k, &err)) return res;
            if (!runNamed(part_[k], {{"x", h.data()}, {"emb", embN.data()}, {"lora", lora.data()}, {"ctx", ctx.data()},
                                     {"cemb", cemb.data()}, {"st", st.data()}},
                          {{"x_out", h2.data()}}, &err)) return res;
            done(k);
            std::swap(h, h2);
        }
        if (!loadPart(3, &err)) return res;
        if (!runNamed(part_[3], {{"x", h.data()}, {"emb", embN.data()}, {"lora", lora.data()}, {"ctx", ctx.data()},
                                 {"cemb", cemb.data()}, {"st", st.data()}},
                      {{"out", out.data()}}, &err)) return res;
        done(3);
        unpatchify(out.data(), v.data(), hp, wp);
        for (size_t k = 0; k < NL; k++) lat[k] += (sn - s) * v[k];  // Euler on the flow velocity
        progress(i + 1, req.steps);
    }
    double t3 = nowMs();
    res.msDraw = t3 - t2;

    const int N = W * Hh;
    std::vector<float> rgb(3 * (size_t)N);
    if (!runNamed(dec_, {{"lat", lat.data()}}, {{"rgb", rgb.data()}}, &err)) return res;
    res.argb.resize(N);
    for (int p = 0; p < N; p++) res.argb[p] = argb(rgb[p], rgb[N + p], rgb[2 * (size_t)N + p]);
    res.msDecode = nowMs() - t3;
    res.latent = std::move(lat);
    return res;
}

// XLSR is compiled for 512x512 -> 2048x2048, so bigger or non-square pictures go through in tiles that overlap
// by 2*O pixels; each tile keeps its own centre, so the seams come from well inside a tile. Pictures smaller
// than a tile are padded by repeating the edge.
std::vector<uint32_t> Pipeline::upscale(const std::vector<uint32_t>& in, int w, int h, std::string* err, double* ms) {
    double t0 = nowMs();
    constexpr int T = 512, O = 16, S = T - 2 * O, U = 4, TU = T * U;
    if (w <= 0 || h <= 0 || in.size() != (size_t)w * h) { *err = "upscale: wrong picture size"; return {}; }
    QnnGraph up;
    if (!up.load(dir_ + "/xlsr.bin", err)) return {};
    auto starts = [&](int n) {
        std::vector<int> v;
        if (n <= T) { v.push_back(0); return v; }
        for (int x = 0;; x += S) {
            if (x + T >= n) { v.push_back(n - T); break; }
            v.push_back(x);
        }
        return v;
    };
    const int N = T * T, M = TU * TU, OW = w * U, OH = h * U;
    std::vector<float> img(3 * N), big(3 * (size_t)M);
    std::vector<uint32_t> outPx((size_t)OW * OH);
    for (int ty : starts(h))
        for (int tx : starts(w)) {
            for (int y = 0; y < T; y++)
                for (int x = 0; x < T; x++) {
                    uint32_t c = in[(size_t)std::min(ty + y, h - 1) * w + std::min(tx + x, w - 1)];
                    int p = y * T + x;
                    img[p] = ((c >> 16) & 255) / 255.f;
                    img[N + p] = ((c >> 8) & 255) / 255.f;
                    img[2 * N + p] = (c & 255) / 255.f;
                }
            if (!runNamed(up, {{"image", img.data()}}, {{"upscaled_image", big.data()}}, err)) return {};
            // this tile writes from its left/top margin on (later tiles overwrite our far margin with their centre)
            int x0 = tx > 0 ? tx + O : 0, y0 = ty > 0 ? ty + O : 0, x1 = std::min(w, tx + T), y1 = std::min(h, ty + T);
            for (int oy = y0 * U; oy < y1 * U; oy++)
                for (int ox = x0 * U; ox < x1 * U; ox++) {
                    size_t q = (size_t)(oy - ty * U) * TU + (ox - tx * U);
                    outPx[(size_t)oy * OW + ox] = argb(big[q], big[M + q], big[2 * (size_t)M + q]);
                }
        }
    *ms = nowMs() - t0;
    return outPx;
}
