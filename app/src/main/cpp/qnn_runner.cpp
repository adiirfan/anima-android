#include "qnn_runner.h"

std::atomic<uint64_t> g_npuBusyNs{0};

#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpSystemContext.h"
#include "QnnTypeMacros.hpp"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "anima", __VA_ARGS__)

typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t*** providers, uint32_t* n);
typedef Qnn_ErrorHandle_t (*GetSysProvidersFn)(const QnnSystemInterface_t*** providers, uint32_t* n);

int g_loadOptions = 7;  // measured 2026-10-07: loading 2.7 s -> 0.8 s, Qwen encoder 1.1 s -> 0.35 s

QnnBackend& QnnBackend::get() {
    static QnnBackend b;
    return b;
}

bool QnnBackend::open(const std::string& libDir, std::string* err) {
    if (backend_) return true;
    // The neural unit's own half of the library (the "skel") is looked up through this path.
    std::string adsp = libDir + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);

    void* lib = dlopen((libDir + "/libQnnHtp.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { *err = std::string("libQnnHtp.so: ") + dlerror(); return false; }
    auto getProviders = (GetProvidersFn)dlsym(lib, "QnnInterface_getProviders");
    const QnnInterface_t** prov = nullptr;
    uint32_t n = 0;
    if (!getProviders || getProviders(&prov, &n) != QNN_SUCCESS || n == 0) { *err = "no QNN interface"; return false; }
    bool found = false;
    for (uint32_t i = 0; i < n; i++) {
        if (prov[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
            prov[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
            qnn_ = prov[i]->QNN_INTERFACE_VER_NAME;
            found = true;
            break;
        }
    }
    if (!found) { *err = "QNN interface version mismatch"; return false; }

    void* slib = dlopen((libDir + "/libQnnSystem.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!slib) { *err = std::string("libQnnSystem.so: ") + dlerror(); return false; }
    auto getSys = (GetSysProvidersFn)dlsym(slib, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** sprov = nullptr;
    if (!getSys || getSys(&sprov, &n) != QNN_SUCCESS || n == 0) { *err = "no QNN system interface"; return false; }
    found = false;
    for (uint32_t i = 0; i < n; i++) {
        if (sprov[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
            sprov[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
            sys_ = sprov[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
            found = true;
            break;
        }
    }
    if (!found) { *err = "QNN system interface version mismatch"; return false; }

    if (qnn_.backendCreate(nullptr, nullptr, &backend_) != QNN_SUCCESS) { *err = "backendCreate failed"; backend_ = nullptr; return false; }
    if (qnn_.deviceCreate && qnn_.deviceCreate(nullptr, nullptr, &device_) != QNN_SUCCESS) { *err = "deviceCreate failed"; return false; }
    setBurst();
    LOGI("QNN backend open (%s)", libDir.c_str());
    return true;
}

// Same as qnn-net-run's "burst" profile: no clock scaling, highest voltage corners, no sleep.
void QnnBackend::setBurst() {
    QnnDevice_Infrastructure_t infra = nullptr;
    if (!qnn_.deviceGetInfrastructure || qnn_.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return;
    auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
    if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    QnnHtpDevice_PerfInfrastructure_t& perf = htp->perfInfra;
    uint32_t id = 0;
    if (perf.createPowerConfigId(0, 0, &id) != QNN_SUCCESS) return;

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& d = dcvs.dcvsV3Config;
    d.contextId = id;
    d.setDcvsEnable = 1; d.dcvsEnable = 0;
    d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    d.setSleepLatency = 1; d.sleepLatency = 40;
    d.setSleepDisable = 1; d.sleepDisable = 1;
    d.setBusParams = 1;
    d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    d.setCoreParams = 1;
    d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    QnnHtpPerfInfrastructure_PowerConfig_t lat = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    lat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    lat.rpcControlLatencyConfig = 100;
    QnnHtpPerfInfrastructure_PowerConfig_t poll = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    poll.rpcPollingTimeConfig = 9999;

    const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, &lat, &poll, nullptr};
    if (perf.setPowerConfig(id, cfgs) == QNN_SUCCESS) LOGI("neural unit set to burst");
}

static bool readFile(const std::string& path, std::vector<uint8_t>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    out->resize((size_t)f.tellg());
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out->data()), (std::streamsize)out->size());
    return (bool)f;
}

// The graph list of a context binary's description.
static const QnnSystemContext_GraphInfo_t* graphList(const QnnSystemContext_BinaryInfo_t* info, uint32_t* n) {
    switch (info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: *n = info->contextBinaryInfoV1.numGraphs; return info->contextBinaryInfoV1.graphs;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: *n = info->contextBinaryInfoV2.numGraphs; return info->contextBinaryInfoV2.graphs;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: *n = info->contextBinaryInfoV3.numGraphs; return info->contextBinaryInfoV3.graphs;
        default: *n = 0; return nullptr;
    }
}

static const char* graphNameOf(const QnnSystemContext_GraphInfo_t& g) {
    if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) return g.graphInfoV1.graphName;
    if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) return g.graphInfoV2.graphName;
    return g.graphInfoV3.graphName;
}

uint64_t QnnGraph::spillSize(const std::string& path, const std::string& graph) {
    QnnBackend& b = QnnBackend::get();
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return 0;
    size_t n = (size_t)lseek(fd, 0, SEEK_END);
    void* m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return 0;
    uint64_t spill = 0;
    QnnSystemContext_Handle_t sys = nullptr;
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t infoSize = 0;
    if (b.sys().systemContextCreate(&sys) == QNN_SUCCESS &&
        b.sys().systemContextGetBinaryInfo(sys, m, n, &info, &infoSize) == QNN_SUCCESS && info) {
        // Newer binaries keep it per graph (graph info V3 blob), older ones in the context's hw info blob.
        uint32_t ng = 0;
        const QnnSystemContext_GraphInfo_t* graphs = graphList(info, &ng);
        for (uint32_t i = 0; graphs && i < ng; i++) {
            const auto& gi = graphs[i];
            if (!graph.empty() && graph != graphNameOf(gi)) continue;
            if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 && gi.graphInfoV3.graphBlobInfo &&
                gi.graphInfoV3.graphBlobInfoSize >= sizeof(QnnHtpSystemContext_GraphBlobInfo_t)) {
                auto* g = static_cast<const QnnHtpSystemContext_GraphBlobInfo_t*>(gi.graphInfoV3.graphBlobInfo);
                if (g->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1)
                    spill = std::max<uint64_t>(spill, g->contextBinaryGraphBlobInfoV1.spillFillBufferSize);
            }
        }
        if (!spill && info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2 && info->contextBinaryInfoV2.hwInfoBlob &&
            info->contextBinaryInfoV2.hwInfoBlobSize >= sizeof(QnnHtpSystemContext_HwBlobInfo_t))
            spill = static_cast<const QnnHtpSystemContext_HwBlobInfo_t*>(info->contextBinaryInfoV2.hwInfoBlob)
                        ->contextBinaryHwInfoBlobV1_t.spillFillBufferSize;
    }
    if (sys) b.sys().systemContextFree(sys);
    munmap(m, n);
    return spill;
}

std::vector<std::string> QnnGraph::graphNames(const std::string& path) {
    std::vector<std::string> names;
    QnnBackend& b = QnnBackend::get();
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return names;
    size_t n = (size_t)lseek(fd, 0, SEEK_END);
    void* m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return names;
    QnnSystemContext_Handle_t sys = nullptr;
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t infoSize = 0;
    if (b.sys().systemContextCreate(&sys) == QNN_SUCCESS &&
        b.sys().systemContextGetBinaryInfo(sys, m, n, &info, &infoSize) == QNN_SUCCESS && info) {
        uint32_t ng = 0;
        const QnnSystemContext_GraphInfo_t* graphs = graphList(info, &ng);
        for (uint32_t i = 0; graphs && i < ng; i++) names.emplace_back(graphNameOf(graphs[i]));
    }
    if (sys) b.sys().systemContextFree(sys);
    munmap(m, n);
    return names;
}

bool QnnGraph::load(const std::string& path, std::string* err, bool group, Qnn_ContextHandle_t groupFirst,
                    uint64_t groupSpill, const std::string& graph) {
    release();
    QnnBackend& b = QnnBackend::get();
    if (!b.isOpen()) { *err = "backend not open"; return false; }
    std::vector<uint8_t> buf;
    const uint8_t* data = nullptr;
    size_t size = 0;
    if (g_loadOptions & 1) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { *err = "cannot open " + path; return false; }
        mapSize_ = (size_t)lseek(fd, 0, SEEK_END);
        map_ = mmap(nullptr, mapSize_, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map_ == MAP_FAILED) { map_ = nullptr; *err = "cannot map " + path; return false; }
        data = static_cast<const uint8_t*>(map_);
        size = mapSize_;
    } else {
        if (!readFile(path, &buf)) { *err = "cannot read " + path; return false; }
        data = buf.data();
        size = buf.size();
    }

    QnnSystemContext_Handle_t sys = nullptr;
    if (b.sys().systemContextCreate(&sys) != QNN_SUCCESS) { *err = "systemContextCreate failed"; return false; }
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t infoSize = 0;
    if (b.sys().systemContextGetBinaryInfo(sys, const_cast<uint8_t*>(data), size, &info, &infoSize) != QNN_SUCCESS || !info) {
        b.sys().systemContextFree(sys);
        *err = "not a context binary: " + path;
        return false;
    }
    uint32_t ng = 0;
    const QnnSystemContext_GraphInfo_t* graphs = graphList(info, &ng);
    if (!graphs || ng == 0) { b.sys().systemContextFree(sys); *err = "no graphs in " + path; return false; }
    uint32_t gi = 0;
    if (!graph.empty()) {
        while (gi < ng && graph != graphNameOf(graphs[gi])) gi++;
        if (gi == ng) {
            *err = "no graph " + graph + " in " + path + " (has:";
            for (uint32_t i = 0; i < ng; i++) *err += std::string(" ") + graphNameOf(graphs[i]);
            *err += ")";
            b.sys().systemContextFree(sys);
            return false;
        }
    }

    const char* gname = nullptr;
    uint32_t nIn = 0, nOut = 0;
    const Qnn_Tensor_t *ins = nullptr, *outs = nullptr;
#define TAKE(V) gname = graphs[gi].V.graphName; nIn = graphs[gi].V.numGraphInputs; ins = graphs[gi].V.graphInputs; \
                nOut = graphs[gi].V.numGraphOutputs; outs = graphs[gi].V.graphOutputs;
    if (graphs[gi].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) { TAKE(graphInfoV1) }
    else if (graphs[gi].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) { TAKE(graphInfoV2) }
    else { TAKE(graphInfoV3) }
#undef TAKE
    std::string graphName = gname ? gname : "";

    // Copy the tensor descriptions before the system context (which owns them) is freed. Reserve so
    // the stored dims/names never move.
    dimStore_.clear(); nameStore_.clear();
    dimStore_.reserve(nIn + nOut); nameStore_.reserve(nIn + nOut);
    auto copy = [&](const Qnn_Tensor_t& src, std::vector<Qnn_Tensor_t>* dst, std::vector<IoInfo>* io) {
        Qnn_Tensor_t t = src;
        uint32_t rank = QNN_TENSOR_GET_RANK(src);
        const uint32_t* d = QNN_TENSOR_GET_DIMENSIONS(src);
        dimStore_.emplace_back(d, d + rank);
        nameStore_.emplace_back(QNN_TENSOR_GET_NAME(src));
        QNN_TENSOR_SET_DIMENSIONS(t, dimStore_.back().data());
        QNN_TENSOR_SET_NAME(t, nameStore_.back().c_str());
        if (t.version == QNN_TENSOR_VERSION_2) t.v2.isDynamicDimensions = nullptr;
        QNN_TENSOR_SET_MEM_TYPE(t, QNN_TENSORMEMTYPE_RAW);
        IoInfo i;
        i.name = nameStore_.back();
        i.dims = dimStore_.back();
        i.count = 1;
        for (uint32_t v : i.dims) i.count *= v;
        i.type = QNN_TENSOR_GET_DATA_TYPE(src);
        dst->push_back(t);
        io->push_back(i);
    };
    for (uint32_t i = 0; i < nIn; i++) copy(ins[i], &inT_, &in_);
    for (uint32_t i = 0; i < nOut; i++) copy(outs[i], &outT_, &out_);
    b.sys().systemContextFree(sys);

    QnnHtpContext_CustomConfig_t accel{}, parallel{};
    accel.option = QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION;
    accel.initAcceleration = true;
    parallel.option = QNN_HTP_CONTEXT_CONFIG_OPTION_PARALLEL_WEIGHT_LOADING;
    parallel.parallelWeightLoading.mode = QNN_HTP_PARALLEL_WEIGHT_LOADING_MODE_ON;
    QnnContext_Config_t c1 = QNN_CONTEXT_CONFIG_INIT, c2 = QNN_CONTEXT_CONFIG_INIT, c3 = QNN_CONTEXT_CONFIG_INIT;
    c1.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c1.customConfig = &accel;
    c2.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c2.customConfig = &parallel;
    c3.option = QNN_CONTEXT_CONFIG_PERSISTENT_BINARY; c3.isPersistentBinary = 1;
    QnnHtpContext_CustomConfig_t grp{};
    grp.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    grp.groupRegistration.firstGroupHandle = groupFirst;
    grp.groupRegistration.maxSpillFillBuffer = groupSpill;
    QnnContext_Config_t c4 = QNN_CONTEXT_CONFIG_INIT;
    c4.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM; c4.customConfig = &grp;
    // Only the chosen picture size is unpacked when the file holds several.
    const char* only[] = {graphName.c_str(), nullptr};
    QnnContext_Config_t c5 = QNN_CONTEXT_CONFIG_INIT;
    c5.option = QNN_CONTEXT_CONFIG_ENABLE_GRAPHS; c5.enableGraphs = only;
    std::vector<const QnnContext_Config_t*> cfg;
    if (ng > 1) cfg.push_back(&c5);
    if (group) cfg.push_back(&c4);
    if (g_loadOptions & 2) cfg.push_back(&c1);
    if (g_loadOptions & 4) cfg.push_back(&c2);
    if (g_loadOptions & 1) cfg.push_back(&c3);
    cfg.push_back(nullptr);
    if (b.qnn().contextCreateFromBinary(b.backend(), b.device(), cfg.data(), data, size, &ctx_, nullptr) != QNN_SUCCESS) {
        ctx_ = nullptr;
        *err = "contextCreateFromBinary failed: " + path;
        return false;
    }
    if (b.qnn().graphRetrieve(ctx_, graphName.c_str(), &graph_) != QNN_SUCCESS) {
        graph_ = nullptr;
        *err = "graphRetrieve failed: " + graphName;
        return false;
    }
    auto bytes = [](const IoInfo& i) { return i.count * (i.type == QNN_DATATYPE_FLOAT_16 ? 2 : 4); };
    for (auto& i : in_) inBuf_.emplace_back(bytes(i));
    for (auto& o : out_) outBuf_.emplace_back(bytes(o));
    LOGI("loaded %s (graph %s, %u in, %u out)", path.c_str(), graphName.c_str(), nIn, nOut);
    return true;
}

void QnnGraph::release() {
    if (ctx_) QnnBackend::get().qnn().contextFree(ctx_, nullptr);
    ctx_ = nullptr;
    if (map_) munmap(map_, mapSize_);
    map_ = nullptr;
    mapSize_ = 0;
    graph_ = nullptr;
    inT_.clear(); outT_.clear(); in_.clear(); out_.clear();
    inBuf_.clear(); outBuf_.clear(); dimStore_.clear(); nameStore_.clear();
}

bool QnnGraph::run(const std::vector<const float*>& in, const std::vector<float*>& out, std::string* err) {
    if (!graph_) { *err = "graph not loaded"; return false; }
    if (in.size() != in_.size() || out.size() != out_.size()) { *err = "wrong number of inputs/outputs"; return false; }
    for (size_t i = 0; i < in_.size(); i++) {
        if (in_[i].type == QNN_DATATYPE_FLOAT_16) {
            auto* h = reinterpret_cast<_Float16*>(inBuf_[i].data());
            for (size_t k = 0; k < in_[i].count; k++) h[k] = (_Float16)in[i][k];
        } else {
            memcpy(inBuf_[i].data(), in[i], in_[i].count * 4);
        }
        QNN_TENSOR_SET_CLIENT_BUF(inT_[i], (Qnn_ClientBuffer_t{inBuf_[i].data(), (uint32_t)inBuf_[i].size()}));
    }
    for (size_t i = 0; i < out_.size(); i++)
        QNN_TENSOR_SET_CLIENT_BUF(outT_[i], (Qnn_ClientBuffer_t{outBuf_[i].data(), (uint32_t)outBuf_[i].size()}));
    Qnn_ErrorHandle_t e;
    {
        NpuBusy busy;
        e = QnnBackend::get().qnn().graphExecute(graph_, inT_.data(), (uint32_t)inT_.size(),
                                                 outT_.data(), (uint32_t)outT_.size(), nullptr, nullptr);
    }
    if (e != QNN_SUCCESS) { *err = "graphExecute failed: " + std::to_string((long)e); return false; }
    for (size_t i = 0; i < out_.size(); i++) {
        if (out_[i].type == QNN_DATATYPE_FLOAT_16) {
            auto* h = reinterpret_cast<const _Float16*>(outBuf_[i].data());
            for (size_t k = 0; k < out_[i].count; k++) out[i][k] = (float)h[k];
        } else {
            memcpy(out[i], outBuf_[i].data(), out_[i].count * 4);
        }
    }
    return true;
}

std::string QnnGraph::describe() const {
    std::ostringstream s;
    auto one = [&](const IoInfo& i) {
        s << "  " << i.name << " [";
        for (size_t k = 0; k < i.dims.size(); k++) s << (k ? "," : "") << i.dims[k];
        s << "] " << (i.type == QNN_DATATYPE_FLOAT_16 ? "fp16" : i.type == QNN_DATATYPE_FLOAT_32 ? "fp32" : "other") << "\n";
    };
    s << "inputs:\n";
    for (auto& i : in_) one(i);
    s << "outputs:\n";
    for (auto& o : out_) one(o);
    return s.str();
}
