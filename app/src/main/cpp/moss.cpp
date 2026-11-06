#include "moss.h"

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

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "moss", __VA_ARGS__)

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

// MOSS's rotation: four position axes (r, f, h, w) of 16 angle pairs each, theta 10000, each angle written twice
// (the two halves of a head are rotated together). pos4: n rows of 4 positions.
void rope(const double* pos4, int n, _Float16* P, _Float16* Q) {
    for (int r = 0; r < n; r++)
        for (int a = 0; a < 4; a++)
            for (int i = 0; i < 16; i++) {
                int k = a * 16 + i;
                double ang = pos4[r * 4 + a] * std::pow(10000.0, -2.0 * i / 32.0);
                P[r * 128 + k] = P[r * 128 + 64 + k] = (_Float16)std::cos(ang);
                Q[r * 128 + k] = Q[r * 128 + 64 + k] = (_Float16)std::sin(ang);
            }
}

// The block streams xi<k>, xt<k> (k = 1..42) alternate between two buffers each: block k reads <k % 2> and writes
// <(k + 1) % 2>. Every other name keeps its own buffer.
std::string streamKey(const std::string& name) {
    if (name.size() < 3 || name[0] != 'x' || (name[1] != 'i' && name[1] != 't')) return name;
    for (size_t i = 2; i < name.size(); i++) if (name[i] < '0' || name[i] > '9') return name;
    return name.substr(0, 2) + std::to_string(atoi(name.c_str() + 2) % 2);
}

double noiseScaler(double x) { return x * (std::exp(std::pow(x, 0.3)) + 10.0); }

}  // namespace

Moss::~Moss() { release(); }

void Moss::release() {
    for (auto& b : bufs_) g_rpc.free(b.second.p);
    bufs_.clear();
}

Shm& Moss::buf(const std::string& key, size_t bytes, std::string* err) {
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

bool Moss::bindAll(QnnCtx& c, std::string* err, const std::string& prefix) {
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

bool Moss::runFile(const std::string& name, std::string* err, const std::string& prefix) {
    QnnCtx c;
    if (!c.load(dir_ + "/" + name, err) || !bindAll(c, err, prefix)) return false;
    if (c.graphs().empty()) { *err = name + " has no graph"; return false; }
    return c.exec(c.graphs()[0], err);
}

bool Moss::init(const std::string& dir, const std::string& libDir, std::string* err) {
    if (ready_) return true;
    dir_ = dir;
    if (const char* a = getenv("MOSS_AHEAD")) ahead_ = std::max(1, atoi(a));
    if (!QnnBackend::get().open(libDir, err)) return false;
    if (!g_rpc.open()) { *err = "libcdsprpc.so not available"; return false; }
    for (const char* f : {"mtxt.bin", "mtime.bin", "md0.bin", "md41.bin", "fdec.bin", "sigmas.txt"})
        if (access((dir + "/" + f).c_str(), R_OK) != 0) { *err = std::string("MOSS file missing: ") + f; return false; }
    if (FILE* f = fopen((dir + "/sampler.txt").c_str(), "r")) {
        char s[32] = {0};
        if (fscanf(f, "%31s", s) == 1) euler_ = std::string(s) == "euler";
        fclose(f);
    }
    LOGI("MOSS ready, %s", euler_ ? "8-step add-on build (Euler, no guidance)" : "25-step ER-SDE with guidance");
    ready_ = true;
    return true;
}

bool Moss::generate(const float* cond, int nc, const float* unc, int nu, uint64_t seed, MossResult* r,
                    std::string* err, const std::function<void(int, int)>& progress) {
    std::vector<double> sig;
    if (FILE* f = fopen((dir_ + "/sigmas.txt").c_str(), "r")) {
        double v;
        while (fscanf(f, "%lf", &v) == 1) if (v > 0 && v <= 1) sig.push_back(v);
        fclose(f);
    }
    if (sig.empty()) { *err = "no noise levels in sigmas.txt"; return false; }
    const int STEPS = (int)sig.size();
    sig.push_back(0);
    const int NP = euler_ ? 1 : 2;   // passes per step: the prompt, and the empty prompt for guidance
    if (nc < 1 || nc > T || (NP == 2 && (nu < 1 || nu > T))) { *err = "prompt too long (more than 256 words with the template)"; return false; }
    if (progress) progress(0, STEPS);
    const int hh = S / 16, ww = S / 16, N = hh * ww;
    const char* pass[2] = {"c_", "u_"};
    const float* rows[2] = {cond, unc};
    const int nrows[2] = {nc, nu};

    // ---- the prompts: rotation tables per pass (the picture sits after the prompt's real words on the word axis),
    // the model's prompt rows from the reader's rows (kept: every step starts from them)
    double t0 = nowMs();
    std::vector<_Float16> xt0[2];
    for (int p = 0; p < NP; p++) {
        std::string pf = pass[p];
        buf(pf + "ctx", (size_t)T * TXT * 2, err); buf(pf + "kbt", (size_t)T * 2, err);
        buf(pf + "pt", (size_t)T * HD * 2, err); buf(pf + "qt", (size_t)T * HD * 2, err);
        buf(pf + "pi", (size_t)N * HD * 2, err); buf(pf + "qi", (size_t)N * HD * 2, err);
        if (!err->empty()) return false;
        auto* x = (_Float16*)bufs_[pf + "ctx"].p;
        memset(x, 0, (size_t)T * TXT * 2);
        for (size_t k = 0; k < (size_t)nrows[p] * TXT; k++) x[k] = (_Float16)rows[p][k];
        auto* kb = (_Float16*)bufs_[pf + "kbt"].p;
        for (int j = 0; j < T; j++) kb[j] = j < nrows[p] ? (_Float16)0.f : NEG;
        std::vector<double> tp((size_t)T * 4, 0.0), ip((size_t)N * 4, 0.0);
        for (int j = 0; j < T; j++) tp[j * 4 + 1] = j;
        for (int y = 0; y < hh; y++)
            for (int xx = 0; xx < ww; xx++) {
                size_t q = (size_t)y * ww + xx;
                ip[q * 4 + 1] = nrows[p];
                ip[q * 4 + 2] = y - (hh - 1) / 2.0;
                ip[q * 4 + 3] = xx - (ww - 1) / 2.0;
            }
        rope(tp.data(), T, (_Float16*)bufs_[pf + "pt"].p, (_Float16*)bufs_[pf + "qt"].p);
        rope(ip.data(), N, (_Float16*)bufs_[pf + "pi"].p, (_Float16*)bufs_[pf + "qi"].p);
        if (!runFile("mtxt.bin", err, pf)) return false;
        Shm& o = bufs_[pf + "xt0"];
        xt0[p].assign((const _Float16*)o.p, (const _Float16*)o.p + o.bytes / 2);
        auto it = bufs_.find(pf + "ctx");   // the reader's rows were read; free them
        g_rpc.free(it->second.p); bufs_.erase(it);
    }
    r->msText = nowMs() - t0;

    // ---- drawing: 25 steps; every step loads the 42 block files in turn and runs each for both passes
    t0 = nowMs();
    std::vector<float> z((size_t)N * C), d((size_t)N * C), old((size_t)N * C), oldD((size_t)N * C);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : z) v = nd(rng);
    // ER-SDE (stage 3) on the flow model, as ComfyUI's sample_er_sde: lambda = sigma / (1 - sigma)
    std::vector<double> s2 = sig;
    if (!euler_ && s2[0] >= 1.0) s2[0] = 0.99997997;   // offset_first_sigma_for_snr at shift 5
    std::vector<double> lam(s2.size());
    for (size_t i = 0; i < s2.size(); i++) lam[i] = s2[i] > 0 ? s2[i] / (1 - s2[i]) : 0.0;
    std::vector<std::string> paths;
    for (int k = 0; k < NB; k++) paths.push_back(dir_ + "/md" + std::to_string(k) + ".bin");
    for (int i = 0; i < STEPS; i++) {
        if (progress) progress(i + 1, STEPS);
        // the step's shared modulation (both passes use the same)
        {
            Shm& ts = buf("c_ts", 256 * 2, err);
            if (!err->empty()) return false;
            auto* t = (_Float16*)ts.p;
            for (int k = 0; k < 128; k++) {
                double a = s2[i] * 1000.0 * std::exp(-std::log(10000.0) * k / 128.0);
                t[k] = (_Float16)std::cos(a); t[128 + k] = (_Float16)std::sin(a);
            }
            if (!runFile("mtime.bin", err, "c_")) return false;
            for (const char* n : {"mod", "head"}) {
                if (NP == 1) break;
                Shm& c = bufs_[std::string("c_") + n];
                Shm& u = buf(std::string("u_") + n, c.bytes, err);
                if (!err->empty()) return false;
                memcpy(u.p, c.p, c.bytes);
            }
        }
        for (int p = 0; p < NP; p++) {
            std::string pf = pass[p];
            Shm& lat = buf(pf + "lat", (size_t)N * C * 2, err);
            Shm& x0 = buf(pf + "xt0", xt0[p].size() * 2, err);
            if (!err->empty()) return false;
            auto* l = (_Float16*)lat.p;
            for (size_t k = 0; k < z.size(); k++) l[k] = (_Float16)z[k];
            memcpy(x0.p, xt0[p].data(), xt0[p].size() * 2);
        }
        const bool slg = !euler_ && s2[i] >= SLG_END;
        // blocks: load ahead on helper threads, run both passes as each one is ready
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
            for (int p = 0; p < NP && ok; p++) {
                std::string pf = pass[p];
                if (p == 1 && slg && k == SLG_BLOCK) {   // skip-layer guidance: the empty-prompt pass leaves this block out
                    for (const char* s : {"xi", "xt"}) {
                        Shm& a = bufs_[pf + s + std::to_string(k % 2)];
                        Shm& b = buf(pf + s + std::to_string((k + 1) % 2), a.bytes, err);
                        if (!err->empty()) { ok = false; break; }
                        memcpy(b.p, a.p, a.bytes);
                    }
                    continue;
                }
                if (!bindAll(*ctxs[k], err, pf) || !ctxs[k]->exec(ctxs[k]->graphs()[0], err)) ok = false;
            }
            ctxs[k].reset();
        }
        for (int k = 0; k < NB; k++) if (loads[k].valid()) loads[k].wait();
        if (!ok) return false;
        auto* vc = (const _Float16*)bufs_["c_vel"].p;
        if (euler_) {   // the 8-step build: one flow step on the prompt's velocity, no guidance
            const float dt = (float)(sig[i + 1] - sig[i]);
            for (size_t k = 0; k < z.size(); k++) z[k] += dt * (float)vc[k];
            LOGI("step %d of %d done", i + 1, STEPS);
            continue;
        }
        // guidance, then the sampler step
        auto* vu = (const _Float16*)bufs_["u_vel"].p;
        for (size_t k = 0; k < z.size(); k++) {
            float u = (float)vu[k];
            d[k] = z[k] - (float)s2[i] * (u + (float)CFG * ((float)vc[k] - u));
        }
        const int stage = std::min(3, i + 1);
        if (sig[i + 1] == 0) {
            z = d;
        } else {
            double ls = lam[i], lt = lam[i + 1];
            double as = s2[i] / ls, at = s2[i + 1] / lt;
            double rr = noiseScaler(lt) / noiseScaler(ls);
            for (size_t k = 0; k < z.size(); k++) z[k] = (float)(at / as * rr) * z[k] + (float)(at * (1 - rr)) * d[k];
            if (stage >= 2) {
                double dt = lt - ls, step = -dt / 200.0, s = 0, su = 0;
                for (int q = 0; q < 200; q++) {
                    double pos = lt + q * step, sc = noiseScaler(pos);
                    s += 1.0 / sc; su += (pos - ls) / sc;
                }
                s *= step; su *= step;
                float c2 = (float)(at * (dt + s * noiseScaler(lt)) / (ls - lam[i - 1]));
                float c3 = stage >= 3 ? (float)(at * (dt * dt / 2 + su * noiseScaler(lt))) : 0.f;
                float den3 = stage >= 3 ? (float)((ls - lam[i - 2]) / 2) : 1.f;
                for (size_t k = 0; k < z.size(); k++) {
                    float dd = (d[k] - old[k]);   // times 1 / (ls - lam[i-1]), folded into c2 below
                    float ddv = dd / (float)(ls - lam[i - 1]);
                    z[k] += c2 * dd;
                    if (stage >= 3) z[k] += c3 * (ddv - oldD[k]) / den3;
                    oldD[k] = ddv;
                }
            }
            float sn = (float)(at * std::sqrt(std::max(0.0, lt * lt - ls * ls * rr * rr)));
            for (auto& v : z) v += sn * nd(rng);
        }
        old = d;
        LOGI("step %d of %d done", i + 1, STEPS);
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
    LOGI("times ms: prompt %.0f steps %.0f (waiting for loads %.0f) decode %.0f", r->msText, r->msSteps, r->msLoadWait, r->msDecode);
    release();
    return true;
}
