#include "gemma.h"

#include <algorithm>

#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include "HTP/QnnHtpContext.h"
#include "QnnTypeMacros.hpp"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "gemma", __VA_ARGS__)

namespace {

double nowMs() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// FastRPC shared memory (the neural unit reads it in place).
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

const QnnSystemContext_GraphInfo_t* graphList(const QnnSystemContext_BinaryInfo_t* info, uint32_t* n) {
    switch (info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: *n = info->contextBinaryInfoV1.numGraphs; return info->contextBinaryInfoV1.graphs;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: *n = info->contextBinaryInfoV2.numGraphs; return info->contextBinaryInfoV2.graphs;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: *n = info->contextBinaryInfoV3.numGraphs; return info->contextBinaryInfoV3.graphs;
        default: *n = 0; return nullptr;
    }
}

size_t typeBytes(Qnn_DataType_t t) {
    switch (t) {
        case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: case QNN_DATATYPE_UFIXED_POINT_32:
        case QNN_DATATYPE_SFIXED_POINT_32: return 4;
        case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_BOOL_8: return 1;
        default: return 2;
    }
}

const void* mapFile(const std::string& path, size_t* n) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    *n = (size_t)lseek(fd, 0, SEEK_END);
    void* m = mmap(nullptr, *n, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return m == MAP_FAILED ? nullptr : m;
}

}  // namespace

// ---------------------------------------------------------------- QnnCtx

bool QnnCtx::parse(const std::string& path, std::string* err) {
    release();
    QnnBackend& b = QnnBackend::get();
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { *err = "cannot open " + path; return false; }
    mapSize_ = (size_t)lseek(fd, 0, SEEK_END);
    map_ = mmap(nullptr, mapSize_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_ == MAP_FAILED) { map_ = nullptr; *err = "cannot map " + path; return false; }

    QnnSystemContext_Handle_t sys = nullptr;
    if (b.sys().systemContextCreate(&sys) != QNN_SUCCESS) { *err = "systemContextCreate failed"; return false; }
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t infoSize = 0;
    if (b.sys().systemContextGetBinaryInfo(sys, map_, mapSize_, &info, &infoSize) != QNN_SUCCESS || !info) {
        b.sys().systemContextFree(sys);
        *err = "not a context binary: " + path;
        return false;
    }
    uint32_t ng = 0;
    const QnnSystemContext_GraphInfo_t* gl = graphList(info, &ng);
    auto copy = [&](const Qnn_Tensor_t& src, std::vector<Qnn_Tensor_t>* dst, std::vector<std::string>* nm, std::vector<size_t>* by) {
        Qnn_Tensor_t t = src;
        uint32_t rank = QNN_TENSOR_GET_RANK(src);
        const uint32_t* d = QNN_TENSOR_GET_DIMENSIONS(src);
        dims_.emplace_back(new std::vector<uint32_t>(d, d + rank));
        names_.emplace_back(new std::string(QNN_TENSOR_GET_NAME(src)));
        QNN_TENSOR_SET_DIMENSIONS(t, dims_.back()->data());
        QNN_TENSOR_SET_NAME(t, names_.back()->c_str());
        if (t.version == QNN_TENSOR_VERSION_2) t.v2.isDynamicDimensions = nullptr;
        size_t n = typeBytes(QNN_TENSOR_GET_DATA_TYPE(src));
        for (uint32_t v : *dims_.back()) n *= v;
        dst->push_back(t);
        nm->push_back(*names_.back());
        by->push_back(n);
    };
    for (uint32_t gi = 0; gi < ng; gi++) {
        Graph g;
        const char* gname = nullptr;
        uint32_t nIn = 0, nOut = 0;
        const Qnn_Tensor_t *ins = nullptr, *outs = nullptr;
#define TAKE(V) gname = gl[gi].V.graphName; nIn = gl[gi].V.numGraphInputs; ins = gl[gi].V.graphInputs; \
                nOut = gl[gi].V.numGraphOutputs; outs = gl[gi].V.graphOutputs;
        if (gl[gi].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) { TAKE(graphInfoV1) }
        else if (gl[gi].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) { TAKE(graphInfoV2) }
        else { TAKE(graphInfoV3) }
#undef TAKE
        g.name = gname ? gname : "";
        for (uint32_t i = 0; i < nIn; i++) copy(ins[i], &g.in, &g.inNames, &g.inBytes);
        for (uint32_t i = 0; i < nOut; i++) copy(outs[i], &g.out, &g.outNames, &g.outBytes);
        graphs_.push_back(std::move(g));
    }
    b.sys().systemContextFree(sys);

    return true;
}

bool QnnCtx::load(const std::string& path, std::string* err, uint64_t groupSpill, Qnn_ContextHandle_t groupFirst,
                  const std::function<bool(const std::string&)>& keep) {
    if (!parse(path, err)) return false;
    std::vector<const char*> only;                     // graphs to deserialise (empty: all)
    if (keep && std::any_of(graphs_.begin(), graphs_.end(), [&](const Graph& g) { return !keep(g.name); })) {
        graphs_.erase(std::remove_if(graphs_.begin(), graphs_.end(), [&](const Graph& g) { return !keep(g.name); }), graphs_.end());
        for (auto& g : graphs_) only.push_back(g.name.c_str());
        only.push_back(nullptr);
    }
    QnnBackend& b = QnnBackend::get();
    QnnContext_Config_t c1 = QNN_CONTEXT_CONFIG_INIT;
    c1.option = QNN_CONTEXT_CONFIG_PERSISTENT_BINARY; c1.isPersistentBinary = 1;
    QnnHtpContext_CustomConfig_t grp{};
    grp.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    grp.groupRegistration.firstGroupHandle = groupFirst;
    grp.groupRegistration.maxSpillFillBuffer = groupSpill;
    QnnContext_Config_t c2 = QNN_CONTEXT_CONFIG_INIT;
    c2.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c2.customConfig = &grp;
    // all neural-unit threads unpack, weights copied in parallel (Anima: loading 2.7 s -> 0.8 s)
    QnnHtpContext_CustomConfig_t accel{}, par{};
    accel.option = QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION; accel.initAcceleration = true;
    par.option = QNN_HTP_CONTEXT_CONFIG_OPTION_PARALLEL_WEIGHT_LOADING; par.parallelWeightLoading.mode = QNN_HTP_PARALLEL_WEIGHT_LOADING_MODE_ON;
    QnnContext_Config_t c3 = QNN_CONTEXT_CONFIG_INIT, c4 = QNN_CONTEXT_CONFIG_INIT;
    c3.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c3.customConfig = &accel;
    c4.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c4.customConfig = &par;
    QnnContext_Config_t c5 = QNN_CONTEXT_CONFIG_INIT;
    c5.option = QNN_CONTEXT_CONFIG_ENABLE_GRAPHS; c5.enableGraphs = only.data();
    std::vector<const QnnContext_Config_t*> cfgv = {&c1, &c3, &c4};
    if (groupSpill) cfgv.push_back(&c2);
    if (!only.empty()) cfgv.push_back(&c5);
    cfgv.push_back(nullptr);
    const QnnContext_Config_t** cfg = cfgv.data();
    if (b.qnn().contextCreateFromBinary(b.backend(), b.device(), cfg, map_, mapSize_, &ctx_, nullptr) != QNN_SUCCESS) {
        ctx_ = nullptr;
        *err = "contextCreateFromBinary failed: " + path;
        return false;
    }
    for (auto& g : graphs_)
        if (b.qnn().graphRetrieve(ctx_, g.name.c_str(), &g.h) != QNN_SUCCESS) { *err = "graphRetrieve failed: " + g.name; return false; }
    LOGI("loaded %s (%zu graphs)", path.c_str(), graphs_.size());
    return true;
}

void QnnCtx::release() {
    auto& q = QnnBackend::get().qnn();
    for (auto& r : reg_) q.memDeRegister(&r.second, 1);
    reg_.clear();
    if (ctx_) q.contextFree(ctx_, nullptr);
    ctx_ = nullptr;
    if (map_) munmap(map_, mapSize_);
    map_ = nullptr;
    graphs_.clear(); dims_.clear(); names_.clear();
}

void QnnCtx::notify(Qnn_ContextHandle_t context, Qnn_GraphHandle_t graph, const char* name,
                    QnnContext_createFromBinaryAsyncNotifyType_t type, void* param, Qnn_ErrorHandle_t status) {
    auto* c = static_cast<QnnCtx*>(param);
    if (type == QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT) c->ctx_ = context;
    else if (type == QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT && name)
        for (auto& g : c->graphs_) if (g.name == name) g.h = graph;
}

bool QnnCtx::loadList(const std::vector<QnnCtx*>& ctxs, const std::vector<std::string>& paths, std::string* err) {
    QnnBackend& b = QnnBackend::get();
    if (!b.qnn().contextCreateFromBinaryListAsync) { *err = "contextCreateFromBinaryListAsync not available"; return false; }
    QnnContext_Config_t persist = QNN_CONTEXT_CONFIG_INIT;
    persist.option = QNN_CONTEXT_CONFIG_PERSISTENT_BINARY; persist.isPersistentBinary = 1;
    const QnnContext_Config_t* perCtx[] = {&persist, nullptr};
    std::vector<std::unique_ptr<QnnContext_Params_t>> params;     // const members: built in place
    std::vector<const QnnContext_Params_t*> plist;
    for (size_t i = 0; i < ctxs.size(); i++) {
        ctxs[i]->release();
        if (!ctxs[i]->parse(paths[i], err)) return false;
        params.emplace_back(new QnnContext_Params_t{
            .version = QNN_CONTEXT_PARAMS_VERSION_1,
            .v1 = QnnContext_ParamsV1_t{perCtx, ctxs[i]->map_, ctxs[i]->mapSize_, nullptr, &QnnCtx::notify, ctxs[i]}});
        plist.push_back(params.back().get());
    }
    plist.push_back(nullptr);
    QnnHtpContext_CustomConfig_t share{}, type{};
    share.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES; share.shareResources = true;
    type.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE; type.shareResOptType = SEQUENTIAL_WITH_VA_OPTIMIZATION;
    QnnContext_Config_t c1 = QNN_CONTEXT_CONFIG_INIT, c2 = QNN_CONTEXT_CONFIG_INIT;
    c1.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c1.customConfig = &share;
    c2.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c2.customConfig = &type;
    const QnnContext_Config_t* listCfg[] = {&c1, &c2, nullptr};
    Qnn_ErrorHandle_t e = b.qnn().contextCreateFromBinaryListAsync(b.backend(), b.device(), plist.data(), listCfg, nullptr);
    if (e != QNN_SUCCESS) { *err = "contextCreateFromBinaryListAsync failed: " + std::to_string((long)e); return false; }
    for (size_t i = 0; i < ctxs.size(); i++) {
        if (!ctxs[i]->ctx_) { *err = "no context for " + paths[i]; return false; }
        for (auto& g : ctxs[i]->graphs_)
            if (!g.h && b.qnn().graphRetrieve(ctxs[i]->ctx_, g.name.c_str(), &g.h) != QNN_SUCCESS) { *err = "graphRetrieve failed: " + g.name; return false; }
    }
    LOGI("loaded %zu files in one shared-resource call", ctxs.size());
    return true;
}

QnnCtx::Graph* QnnCtx::graph(const std::string& part) {
    for (auto& g : graphs_)
        if (g.name.find(part) != std::string::npos) return &g;
    return nullptr;
}

bool QnnCtx::bind(Graph& g, bool input, size_t i, const Shm& m, std::string* err) {
    Qnn_Tensor_t& t = input ? g.in[i] : g.out[i];
    auto key = std::make_pair(m.fd, (input ? g.inBytes : g.outBytes)[i]);
    auto it = reg_.find(key);
    Qnn_MemHandle_t h = nullptr;
    if (it != reg_.end()) {
        h = it->second;
    } else {
        Qnn_MemDescriptor_t d = {{QNN_TENSOR_GET_RANK(t), QNN_TENSOR_GET_DIMENSIONS(t), nullptr},
                                 QNN_TENSOR_GET_DATA_TYPE(t), QNN_MEM_TYPE_ION, {{-1}}};
        d.ionInfo.fd = m.fd;
        if (QnnBackend::get().qnn().memRegister(ctx_, &d, 1, &h) != QNN_SUCCESS) {
            *err = "memRegister failed for " + std::string(QNN_TENSOR_GET_NAME(t));
            return false;
        }
        reg_[key] = h;
    }
    QNN_TENSOR_SET_MEM_TYPE(t, QNN_TENSORMEMTYPE_MEMHANDLE);
    QNN_TENSOR_SET_MEM_HANDLE(t, h);
    return true;
}

std::map<std::string, std::pair<double, int>> g_execMs;  // graph -> (total ms, runs), for profiling

bool QnnCtx::exec(Graph& g, std::string* err) {
    double t0 = nowMs();
    Qnn_ErrorHandle_t e;
    {
        NpuBusy busy;
        e = QnnBackend::get().qnn().graphExecute(g.h, g.in.data(), (uint32_t)g.in.size(), g.out.data(), (uint32_t)g.out.size(),
                                                 nullptr, nullptr);
    }
    if (e != QNN_SUCCESS) { *err = "graphExecute failed (" + g.name + "): " + std::to_string((long)e); return false; }
    auto& a = g_execMs[g.name];
    a.first += nowMs() - t0; a.second++;
    return true;
}

// ---------------------------------------------------------------- Gemma

Gemma::~Gemma() {
    for (auto& c : part_) c.release();
    draft_.release();
    for (auto& b : bufs_) g_rpc.free(b.second.p);
    if (emb_) munmap((void*)emb_, embBytes_);
    if (ple_) munmap((void*)ple_, pleBytes_);
}

Shm Gemma::alloc(size_t bytes) {
    Shm s;
    s.p = g_rpc.alloc(25 /* system heap */, 1 /* default flags */, (int)bytes);
    if (s.p) { s.bytes = bytes; s.fd = g_rpc.toFd(s.p); memset(s.p, 0, bytes); }
    return s;
}

Shm& Gemma::buf(const std::string& key, size_t bytes, std::string* err) {
    auto it = bufs_.find(key);
    if (it != bufs_.end()) {
        if (it->second.bytes != bytes) *err = "buffer " + key + " size mismatch";
        return it->second;
    }
    Shm s = alloc(bytes);
    if (!s.p) *err = "rpcmem_alloc failed for " + key;
    return bufs_[key] = s;
}

// Every tensor gets a buffer named after it: the conversation memory (*_in keys/values) is one buffer for every
// graph; anything else is one buffer per graph size ("name@T"), shared by the three pieces of that size.
bool Gemma::bindAll(QnnCtx& c, std::string* err) {
    for (auto& g : c.graphs()) {
        std::string T = g.name.substr(g.name.find("_ar") + 3);
        T = T.substr(0, T.find('_'));
        for (int io = 0; io < 2; io++) {
            auto& names = io ? g.outNames : g.inNames;
            auto& bytes = io ? g.outBytes : g.inBytes;
            for (size_t i = 0; i < names.size(); i++) {
                const std::string& n = names[i];
                bool mem = !io && n.size() > 3 && n.compare(n.size() - 3, 3, "_in") == 0 &&
                           (n.rfind("swa_", 0) == 0 || n.rfind("past_", 0) == 0);
                Shm& s = buf(mem ? n : n + "@" + T, bytes[i], err);
                if (!err->empty()) return false;
                if (!c.bind(g, io == 0, i, s, err)) return false;
            }
        }
    }
    return true;
}

bool Gemma::init(const std::string& dir, const std::string& libDir, std::string* err) {
    double t0 = nowMs();
    if (!QnnBackend::get().open(libDir, err)) return false;
    if (!g_rpc.open()) { *err = "libcdsprpc.so (shared memory) not available"; return false; }
    for (int i = 0; i < 42; i++) if (i < 24) own_.push_back(i);
    ring_.assign(SW, -1);
    for (int k = 0; k < 3; k++) {
        if (!part_[k].load(dir + "/part" + std::to_string(k + 1) + ".bin", err)) return false;
        for (int T : {128, 4, 1}) {
            QnnCtx::Graph* g = part_[k].graph("_ar" + std::to_string(T) + "_");
            if (!g) { *err = "part" + std::to_string(k + 1) + " has no graph for " + std::to_string(T) + " words"; return false; }
            g_[k][T] = g;
        }
        if (!bindAll(part_[k], err)) return false;
    }
    if (draft_.load(dir + "/draft.bin", err)) {
        gd_ = &draft_.graphs()[0];
        if (!bindAll(draft_, err)) return false;
    } else {
        LOGI("no drafter: %s", err->c_str());
        err->clear();
    }
    emb_ = (const uint16_t*)mapFile(dir + "/embedding_int16_lut.bin", &embBytes_);
    ple_ = (const uint8_t*)mapFile(dir + "/embed_token_int8_lut.bin", &pleBytes_);
    if (!emb_ || !ple_) { *err = "word tables missing"; return false; }
    // scales: luts.json written by the PC tools ({"embedding": {..., "scale", "offset"}, "perlayer_embedding": {...},
    // "perlayer16": {"scale", "offset"} = the graph's 16-bit input encoding})
    std::ifstream f(dir + "/luts.txt");
    float s8 = 0, s16 = 0; int o8 = 0, o16 = 0;
    if (!(f >> embScale_ >> embOff_ >> s8 >> o8 >> s16 >> o16)) { *err = "luts.txt missing"; return false; }
    for (int q = 0; q < 256; q++) {
        float v = (q + o8) * s8;
        long r = lroundf(v / s16 - o16);
        pleTo16_[q] = (uint16_t)std::max(0L, std::min(65535L, r));
    }
    LOGI("gemma ready in %.0f ms", nowMs() - t0);
    return true;
}

void Gemma::setWords(int T, const int32_t* ids, int n) {
    std::string t = "@" + std::to_string(T);
    auto* e = (uint16_t*)bufs_["inputs_embeds" + t].p;
    auto* p = (uint16_t*)bufs_["per_layer_inputs" + t].p;
    for (int r = 0; r < T; r++) {
        int id = r < n ? ids[r] : 0;
        memcpy(e + (size_t)r * D, emb_ + (size_t)id * D, D * 2);
        const uint8_t* src = ple_ + (size_t)id * NL * PLE;
        uint16_t* dst = p + (size_t)r * NL * PLE;
        for (int k = 0; k < NL * PLE; k++) dst[k] = pleTo16_[src[k]];
    }
}

// Rotation tables as Genie builds them: sliding layers theta 1e4 over 256 dims (128 pairs); global layers theta 1e6
// over 512 dims with only the first 64 of 256 pairs rotating ("proportional", partial factor 0.25).
void Gemma::setRope(const std::string& sfx, const int* pos, int n) {
    auto* cs = (_Float16*)bufs_["swa_position_ids_cos" + sfx].p;
    auto* ss = (_Float16*)bufs_["swa_position_ids_sin" + sfx].p;
    auto* cg = (_Float16*)bufs_["position_ids_cos" + sfx].p;
    auto* sg = (_Float16*)bufs_["position_ids_sin" + sfx].p;
    for (int r = 0; r < n; r++) {
        double p = pos[r];
        for (int i = 0; i < 128; i++) {
            double a = p * std::pow(10000.0, -2.0 * i / 256);
            cs[r * 128 + i] = (_Float16)std::cos(a); ss[r * 128 + i] = (_Float16)std::sin(a);
        }
        for (int i = 0; i < 256; i++) {
            double a = i < 64 ? p * std::pow(1000000.0, -2.0 * i / 512) : 0.0;
            cg[r * 256 + i] = (_Float16)std::cos(a); sg[r * 256 + i] = (_Float16)std::sin(a);
        }
    }
}

void Gemma::setMasks(int T, int pos0, int real) {
    std::string t = "@" + std::to_string(T);
    auto* mg = (_Float16*)bufs_["attention_mask" + t].p;
    auto* ms = (_Float16*)bufs_["swa_attention_mask" + t].p;
    const _Float16 NO = (_Float16)-1000.f, YES = (_Float16)0.f;
    for (int r = 0; r < T; r++) {
        int q = pos0 + r;
        _Float16* g = mg + (size_t)r * (SG + T);
        _Float16* s = ms + (size_t)r * (SW + T);
        for (int k = 0; k < SG; k++) g[k] = k < nStored_ ? YES : NO;
        for (int k = 0; k < SW; k++) s[k] = (ring_[k] >= 0 && ring_[k] > q - SW) ? YES : NO;
        for (int j = 0; j < T; j++) {
            bool ok = j <= r && j < real;
            g[SG + j] = ok ? YES : NO;
            s[SW + j] = ok ? YES : NO;
        }
    }
}

// Copies the first `count` new words' keys/values (from this pass's outputs) into the conversation memory.
void Gemma::store(int T, int pos0, int count) {
    std::string t = "@" + std::to_string(T);
    for (int i : own_) {
        bool glob = i % 6 == 5;
        int d = glob ? 512 : 256, S = glob ? SG : SW;
        std::string p = glob ? "past" : "swa";
        for (int h = 0; h < 2; h++) {
            std::string ks = p + "_key_" + std::to_string(i) + "_h" + std::to_string(h);
            std::string vs = p + "_value_" + std::to_string(i) + "_h" + std::to_string(h);
            auto* kn = (const _Float16*)bufs_[ks + "_out" + t].p;   // [d, T]
            auto* vn = (const _Float16*)bufs_[vs + "_out" + t].p;   // [T, d]
            auto* kc = (_Float16*)bufs_[ks + "_in"].p;              // [d, S]
            auto* vc = (_Float16*)bufs_[vs + "_in"].p;              // [S, d]
            for (int c = 0; c < count; c++) {
                int slot = glob ? pos0 + c : (pos0 + c) % SW;
                for (int e = 0; e < d; e++) kc[(size_t)e * S + slot] = kn[(size_t)e * T + c];
                memcpy(vc + (size_t)slot * d, vn + (size_t)c * d, d * 2);
            }
        }
    }
    for (int c = 0; c < count; c++) ring_[(pos0 + c) % SW] = pos0 + c;
    nStored_ = pos0 + count;
}

bool Gemma::pass(int T, std::string* err) {
    for (int k = 0; k < 3; k++)
        if (!part_[k].exec(*g_[k][T], err)) return false;
    return true;
}

static int argmax16(const _Float16* x, int n) {
    int b = 0;
    float m = (float)x[0];
    for (int i = 1; i < n; i++) if ((float)x[i] > m) { m = (float)x[i]; b = i; }
    return b;
}

int Gemma::draft(int t, const _Float16* h, int pos, _Float16* hNext, std::string* err) {
    auto* e = (_Float16*)bufs_["draft_embed@1"].p;
    for (int k = 0; k < D; k++) e[k] = (_Float16)((emb_[(size_t)t * D + k] + embOff_) * embScale_);
    memcpy(bufs_["draft_h@1"].p, h, D * 2);
    setRope("@1", &pos, 1);
    auto* mg = (_Float16*)bufs_["draft_mask@1"].p;
    auto* ms = (_Float16*)bufs_["draft_swa_mask@1"].p;
    for (int k = 0; k < SG; k++) mg[k] = (_Float16)(k < nStored_ ? 0.f : -1000.f);
    for (int k = 0; k < SW; k++) ms[k] = (_Float16)((ring_[k] >= 0 && ring_[k] > pos - SW) ? 0.f : -1000.f);
    if (!draft_.exec(*gd_, err)) return -1;
    memcpy(hNext, bufs_["draft_h_next@1"].p, D * 2);
    return argmax16((const _Float16*)bufs_["draft_logits@1"].p, V);
}

std::vector<int32_t> Gemma::generate(const std::vector<int32_t>& ids, int maxNew, int drafts, GemmaStats* st, std::string* err) {
    std::vector<int32_t> out;
    nStored_ = 0;
    ring_.assign(SW, -1);
    if (drafts > 0 && !gd_) drafts = 0;
    if (drafts > 3) drafts = 3;
    int n = (int)ids.size();
    double t0 = nowMs();
    int last = 0;
    for (int c = 0; c < n; c += 128) {
        int real = std::min(128, n - c);
        setWords(128, ids.data() + c, real);
        std::vector<int> pos(128);
        for (int r = 0; r < 128; r++) pos[r] = c + r;
        setRope("@128", pos.data(), 128);
        setMasks(128, c, real);
        if (!pass(128, err)) return out;
        store(128, c, real);
        last = real - 1;
    }
    auto* lg128 = (const _Float16*)bufs_["logits@128"].p;
    auto* hf128 = (const _Float16*)bufs_["h_final@128"].p;
    int nxt = argmax16(lg128 + (size_t)last * V, V);
    std::vector<_Float16> h(hf128 + (size_t)last * D, hf128 + (size_t)last * D + D), hd(D);
    st->promptWords = n;
    st->msPrompt = nowMs() - t0;
    out.push_back(nxt);

    double t1 = nowMs();
    const int T = drafts > 0 ? 4 : 1;
    std::string t = "@" + std::to_string(T);
    while ((int)out.size() < maxNew && nxt != 1 && nxt != 106) {
        int pos = nStored_;
        std::vector<int32_t> words = {nxt};
        double td = nowMs();
        if (drafts > 0) {
            std::vector<_Float16> hc = h;
            int g = nxt;
            for (int j = 0; j < drafts; j++) {
                g = draft(g, hc.data(), pos + j, hd.data(), err);
                if (g < 0) return out;
                hc = hd;
                words.push_back(g);
            }
            st->msDraft += nowMs() - td;
        }
        double tc = nowMs();
        setWords(T, words.data(), (int)words.size());
        std::vector<int> p(T);
        for (int r = 0; r < T; r++) p[r] = pos + r;
        setRope(t, p.data(), T);
        setMasks(T, pos, (int)words.size());
        if (!pass(T, err)) return out;
        st->msCheck += nowMs() - tc;
        auto* lg = (const _Float16*)bufs_["logits" + t].p;
        auto* hf = (const _Float16*)bufs_["h_final" + t].p;
        int acc = 0;
        int pred = argmax16(lg, V);
        while (acc < drafts && words[acc + 1] == pred) {
            acc++;
            pred = argmax16(lg + (size_t)acc * V, V);
        }
        store(T, pos, acc + 1);
        st->checks++; st->guessed += drafts; st->accepted += acc;
        bool end = false;
        for (int j = 1; j <= acc && !end; j++) {
            out.push_back(words[j]);
            end = words[j] == 1 || words[j] == 106;   // an accepted guess ended the answer
        }
        if (end) break;
        nxt = pred;
        memcpy(h.data(), hf + (size_t)acc * D, D * 2);
        out.push_back(nxt);
    }
    st->msWrite = nowMs() - t1;
    st->newWords = (int)out.size();
    return out;
}
