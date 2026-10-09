#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

// Time spent running graphs on the neural unit (all models), read by the app's usage monitor. NpuBusy adds the
// lifetime of the scope it lives in.
extern std::atomic<uint64_t> g_npuBusyNs;
struct NpuBusy {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~NpuBusy() { g_npuBusyNs += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); }
};

struct IoInfo {
    std::string name;
    std::vector<uint32_t> dims;
    size_t count = 0;
    Qnn_DataType_t type = QNN_DATATYPE_UNDEFINED;
};

// One per process: the neural-unit backend and device, opened once and kept.
class QnnBackend {
public:
    static QnnBackend& get();
    bool open(const std::string& libDir, std::string* err);
    bool isOpen() const { return backend_ != nullptr; }
    const QNN_INTERFACE_VER_TYPE& qnn() const { return qnn_; }
    const QNN_SYSTEM_INTERFACE_VER_TYPE& sys() const { return sys_; }
    Qnn_BackendHandle_t backend() const { return backend_; }
    Qnn_DeviceHandle_t device() const { return device_; }

private:
    void setBurst();
    QNN_INTERFACE_VER_TYPE qnn_{};
    QNN_SYSTEM_INTERFACE_VER_TYPE sys_{};
    Qnn_LogHandle_t log_ = nullptr;
    Qnn_BackendHandle_t backend_ = nullptr;
    Qnn_DeviceHandle_t device_ = nullptr;
};

// How pieces are loaded (bits, set once from the app): 1 = map the file instead of reading it into a copy
// (and tell QNN the buffer stays put), 2 = init acceleration (all DSP threads unpack), 4 = parallel weight copy.
extern int g_loadOptions;

// One compiled piece (context binary with one graph, or several picture sizes sharing their weights). Inputs and outputs are passed as
// float32 and converted to whatever the graph stores (fp16 here).
class QnnGraph {
public:
    ~QnnGraph() { release(); }
    // group: share one scratch (spill-fill) buffer with other pieces. groupFirst = null starts a new
    // group sized groupSpill bytes (must cover every member); otherwise join groupFirst's group.
    // graph: which graph to use when the file holds several (only that one is loaded); "" = the first.
    bool load(const std::string& path, std::string* err, bool group = false,
              Qnn_ContextHandle_t groupFirst = nullptr, uint64_t groupSpill = 0, const std::string& graph = "");
    Qnn_ContextHandle_t context() const { return ctx_; }
    // Scratch (spill-fill) bytes a compiled piece needs, read from its header (the largest over its
    // graphs, or just `graph` when given); 0 if unknown.
    static uint64_t spillSize(const std::string& path, const std::string& graph = "");
    // Names of the graphs in a compiled piece (empty if the file is not readable).
    static std::vector<std::string> graphNames(const std::string& path);
    void release();
    bool loaded() const { return graph_ != nullptr; }
    const std::vector<IoInfo>& inputs() const { return in_; }
    const std::vector<IoInfo>& outputs() const { return out_; }
    // in/out are float32 buffers in the graphs' input/output order, each IoInfo::count long.
    bool run(const std::vector<const float*>& in, const std::vector<float*>& out, std::string* err);
    std::string describe() const;

private:
    Qnn_ContextHandle_t ctx_ = nullptr;
    void* map_ = nullptr;  // the mapped file when loaded with option 1; kept until release
    size_t mapSize_ = 0;
    Qnn_GraphHandle_t graph_ = nullptr;
    std::vector<Qnn_Tensor_t> inT_, outT_;
    std::vector<std::vector<uint32_t>> dimStore_;
    std::vector<std::string> nameStore_;
    std::vector<IoInfo> in_, out_;
    std::vector<std::vector<uint8_t>> inBuf_, outBuf_;
};
