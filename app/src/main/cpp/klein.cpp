#include "klein.h"

#include <android/log.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <random>

#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "anima", __VA_ARGS__)
#include <cstdlib>
#include <fstream>

namespace {

constexpr int T = 128;             // prompt slots (the prompt first, then padding the drawing model never reads)
constexpr int QH = 2560, D = 3072, C = 128;
constexpr int32_t PAD = 151643;
constexpr size_t ROWS = 151936;
constexpr float NEG = -30000.f;

// Debugging: with KLEIN_DUMP=<dir> set, stage outputs are written there as float32 files.
void dump(const char* name, const std::vector<float>& v) {
    const char* d = getenv("KLEIN_DUMP");
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

// Rope tables in the half-split layout (out = x*P + swap(x)*Q), 64 angles per position.
void ropeText(std::vector<float>* P, std::vector<float>* Q) {
    P->resize(T * 128); Q->resize(T * 128);
    for (int p = 0; p < T; p++)
        for (int i = 0; i < 64; i++) {
            double a = p / std::pow(1e6, 2.0 * i / 128.0);
            (*P)[p * 128 + i] = (*P)[p * 128 + 64 + i] = (float)std::cos(a);
            (*Q)[p * 128 + i] = (*Q)[p * 128 + 64 + i] = (float)std::sin(a);
        }
}

// Drawing positions (axis 0 reference index, 1 row, 2 column, 3 text index): 16 frequencies per axis, theta 2000.
void ropeDraw(int h, int w, bool edit, std::vector<float>* P, std::vector<float>* Q) {
    const int Li = h * w, L = T + Li * (edit ? 2 : 1);
    P->resize((size_t)L * 128); Q->resize((size_t)L * 128);
    auto put = [&](int row, const double id[4]) {
        for (int a = 0; a < 4; a++)
            for (int i = 0; i < 16; i++) {
                double ang = id[a] * std::pow(2000.0, -(2.0 * i) / 32.0);
                int k = a * 16 + i;
                (*P)[(size_t)row * 128 + k] = (*P)[(size_t)row * 128 + 64 + k] = (float)std::cos(ang);
                (*Q)[(size_t)row * 128 + k] = (*Q)[(size_t)row * 128 + 64 + k] = (float)std::sin(ang);
            }
    };
    for (int t = 0; t < T; t++) { double id[4] = {0, 0, 0, (double)t}; put(t, id); }
    for (int r = 0; r < (edit ? 2 : 1); r++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                double id[4] = {r ? 10.0 : 0.0, (double)y, (double)x, 0};
                put(T + r * Li + y * w + x, id);
            }
}

// ComfyUI's timestep_embedding(sigma, 256): time factor 1000, [cos | sin].
void temb(float sigma, float* out) {
    for (int i = 0; i < 128; i++) {
        double a = sigma * 1000.0 * std::exp(-std::log(10000.0) * i / 128.0);
        out[i] = (float)std::cos(a);
        out[128 + i] = (float)std::sin(a);
    }
}

}  // namespace

// comfy_extras/nodes_flux.py get_schedule: empirical mu, then t -> e^mu / (e^mu + (1/t - 1)).
std::vector<float> KleinPipeline::sigmas(int steps, int seq) {
    const double a1 = 8.73809524e-05, b1 = 1.89833333, a2 = 0.00016927, b2 = 0.45666666;
    double mu;
    if (seq > 4300) {
        mu = a2 * seq + b2;
    } else {
        double m200 = a2 * seq + b2, m10 = a1 * seq + b1;
        double a = (m200 - m10) / 190.0, b = m200 - 200.0 * a;
        mu = a * steps + b;
    }
    std::vector<float> s;
    for (int i = 0; i <= steps; i++) {
        double t = 1.0 - (double)i / steps;
        s.push_back(t <= 0 ? 0.f : (float)(std::exp(mu) / (std::exp(mu) + (1.0 / t - 1.0))));
    }
    return s;
}

bool KleinPipeline::init(const std::string& modelDir, const std::string& libDir, std::string* err) {
    dir_ = modelDir;
    if (!QnnBackend::get().open(libDir, err)) return false;
    release();
    if (table_) munmap(const_cast<int8_t*>(table_), tableBytes_);
    table_ = static_cast<const int8_t*>(mapFile(dir_ + "/klein_qwen_q8.raw", &tableBytes_));
    if (!table_ || tableBytes_ != ROWS * QH + ROWS * 4) { *err = "klein_qwen_q8.raw missing or wrong size"; return false; }
    scale_ = reinterpret_cast<const float*>(table_ + ROWS * QH);
    // One shared scratch buffer for the decoder and the drawing pieces, sized for the largest graph in the files.
    spill_ = QnnGraph::spillSize(dir_ + "/dec.bin");
    for (int k = 0; k < 4; k++) spill_ = std::max(spill_, QnnGraph::spillSize(dir_ + "/part" + std::to_string(k) + ".bin"));
    w_ = h_ = 0;
    return true;
}

void KleinPipeline::release() {
    for (auto& t : text_) t.release();
    for (auto& p : part_) p.release();
    dec_.release();  // last: it anchors the scratch group
}

bool KleinPipeline::runText(const KleinRequest& req, std::vector<float> taps[3], std::string* err) {
    const int n = (int)req.ids.size();
    if (n < 1 || n > T) { *err = "prompt too long (" + std::to_string(n) + " > " + std::to_string(T) + " pieces)"; return false; }
    std::vector<float> x((size_t)T * QH), P, Q;
    for (int r = 0; r < T; r++) {
        int32_t id = r < n ? req.ids[r] : PAD;
        if (id < 0 || (size_t)id >= ROWS) { *err = "bad Qwen id"; return false; }
        const int8_t* row = table_ + (size_t)id * QH;
        for (int k = 0; k < QH; k++) x[(size_t)r * QH + k] = row[k] * scale_[id];
    }
    ropeText(&P, &Q);
    for (int k = 0; k < 3; k++) {
        if (!text_[k].loaded() && !text_[k].load(dir_ + "/text" + std::to_string(k) + ".bin", err)) return false;
        taps[k].assign((size_t)T * QH, 0.f);
        if (!runNamed(text_[k], {{"x", x.data()}, {"P", P.data()}, {"Q", Q.data()}}, {{"x_out", taps[k].data()}}, err)) return false;
        if (mode_ != 0) text_[k].release();
        x = taps[k];
        // the drawing model reads the padding slots only as masked keys; keep them clean
        std::fill(taps[k].begin() + (size_t)n * QH, taps[k].end(), 0.f);
    }
    return true;
}

bool KleinPipeline::loadPart(int k, std::string* err) {
    if (part_[k].loaded()) return true;
    bool g = spill_ > 0 && dec_.loaded();
    return part_[k].load(dir_ + "/part" + std::to_string(k) + ".bin", err, g, g ? dec_.context() : nullptr, 0, graph_);
}

KleinResult KleinPipeline::generate(const KleinRequest& req, const std::function<void(int, int)>& progress) {
    KleinResult res;
    std::string& err = res.error;
    const int W = req.width, H = req.height, h = H / 16, w = W / 16, Li = h * w;
    const bool edit = !req.edit.empty();
    if (W % 16 || H % 16) { err = "size must be a multiple of 16"; return res; }
    if (edit && req.edit.size() != (size_t)W * H) { err = "picture to edit has the wrong size"; return res; }
    std::string key = std::to_string(W) + "x" + std::to_string(H);
    std::string g = (edit ? "e" : "s") + key;
    if (g != graph_) { release(); graph_ = g; }  // pieces hold one graph each: another size / mode reloads them
    if (mode_ != 0) { for (auto& p : part_) p.release(); dec_.release(); }

    double t0 = nowMs();
    std::vector<float> taps[3];
    if (!runText(req, taps, &err)) return res;
    dump("t9", taps[0]); dump("t18", taps[1]); dump("t27", taps[2]);
    double t1 = nowMs();
    res.msText = t1 - t0;

    // the decoder anchors the scratch group; the encoder is used once, outside it
    if (!dec_.loaded() && !dec_.load(dir_ + "/dec.bin", &err, spill_ > 0, nullptr, spill_, "s" + key)) {
        LOGW("shared scratch failed (%s); loading without it", err.c_str());
        spill_ = 0; err.clear();
        if (!dec_.load(dir_ + "/dec.bin", &err, false, nullptr, 0, "s" + key)) return res;
    }
    std::vector<float> img((size_t)Li * (edit ? 2 : 1) * C);
    if (edit) {
        const int N = W * H;
        std::vector<float> rgb(3 * (size_t)N), lat((size_t)C * Li);
        for (int p = 0; p < N; p++) {
            uint32_t c = req.edit[p];
            rgb[p] = ((c >> 16) & 255) / 255.f; rgb[N + p] = ((c >> 8) & 255) / 255.f; rgb[2 * (size_t)N + p] = (c & 255) / 255.f;
        }
        QnnGraph enc;
        if (!enc.load(dir_ + "/enc.bin", &err, false, nullptr, 0, "s" + key)) return res;
        if (!runNamed(enc, {{"rgb", rgb.data()}}, {{"lat", lat.data()}}, &err)) return res;
        for (int j = 0; j < Li; j++)
            for (int c = 0; c < C; c++) img[((size_t)Li + j) * C + c] = lat[(size_t)c * Li + j];
    }
    for (int k = 0; k < resident(); k++)
        if (!loadPart(k, &err)) return res;
    double t2 = nowMs();
    res.msLoad = t2 - t1;

    const int L = T + Li * (edit ? 2 : 1);
    std::vector<float> P, Q, kb(L, 0.f);
    ropeDraw(h, w, edit, &P, &Q);
    for (int t = (int)req.ids.size(); t < T; t++) kb[t] = NEG;
    std::vector<float> lat((size_t)C * Li), x((size_t)L * D), x2((size_t)L * D), ms(3 * D), mf(2 * D), out((size_t)Li * C), te(256);
    std::mt19937_64 rng(req.seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : lat) v = nd(rng);                 // sigma starts at 1: the latent is the noise itself
    std::vector<float> sig = sigmas(req.steps, Li);
    auto done = [&](int k) { if (k >= resident()) part_[k].release(); };
    for (int i = 0; i < req.steps; i++) {
        float s = sig[i], sn = sig[i + 1];
        temb(s, te.data());
        for (int j = 0; j < Li; j++)
            for (int c = 0; c < C; c++) img[(size_t)j * C + c] = lat[(size_t)c * Li + j];
        if (!loadPart(0, &err)) return res;
        if (!runNamed(part_[0], {{"img", img.data()}, {"t9", taps[0].data()}, {"t18", taps[1].data()}, {"t27", taps[2].data()},
                                 {"temb", te.data()}, {"P", P.data()}, {"Q", Q.data()}, {"kb", kb.data()}},
                      {{"x", x.data()}, {"ms", ms.data()}, {"mf", mf.data()}}, &err)) return res;
        if (i == 0) { dump("img0", img); dump("x0", x); dump("ms", ms); dump("mf", mf); dump("lat0", lat); }
        done(0);
        for (int k = 1; k <= 2; k++) {
            if (!loadPart(k, &err)) return res;
            if (!runNamed(part_[k], {{"x", x.data()}, {"ms", ms.data()}, {"mf", mf.data()}, {"P", P.data()}, {"Q", Q.data()}, {"kb", kb.data()}},
                          {{"x_out", x2.data()}}, &err)) return res;
            done(k);
            std::swap(x, x2);
        }
        if (!loadPart(3, &err)) return res;
        if (!runNamed(part_[3], {{"x", x.data()}, {"ms", ms.data()}, {"mf", mf.data()}, {"P", P.data()}, {"Q", Q.data()}, {"kb", kb.data()}},
                      {{"out", out.data()}}, &err)) return res;
        done(3);
        if (i == 0) dump("out0", out);
        for (int j = 0; j < Li; j++)
            for (int c = 0; c < C; c++) lat[(size_t)c * Li + j] += (sn - s) * out[(size_t)j * C + c];  // Euler, flow velocity
        progress(i + 1, req.steps);
    }
    dump("latN", lat);
    double t3 = nowMs();
    res.msDraw = t3 - t2;

    const int N = W * H;
    std::vector<float> rgb(3 * (size_t)N);
    if (!runNamed(dec_, {{"lat", lat.data()}}, {{"rgb", rgb.data()}}, &err)) return res;
    res.argb.resize(N);
    for (int p = 0; p < N; p++) res.argb[p] = argb(rgb[p], rgb[N + p], rgb[2 * (size_t)N + p]);
    res.msDecode = nowMs() - t3;
    return res;
}
