#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "qnn_runner.h"

// Gemma 4 E4B on the neural unit with our own pieces (gemma-qnn/gemma_ours.py), our own memory handling and the
// MTP drafter. Files in the model folder:
//   part1.bin  layers 0-23 + per-layer-input projection   graphs for 128 / 4 / 1 new words
//   part2.bin  layers 24-41 (reuse layer 22/23 keys/values) same three
//   part3.bin  final norm, word scores, final state          same three
//   draft.bin  MTP drafter (one guess per run)
//   embedding_int16_lut.bin, embed_token_int8_lut.bin, luts.json (word tables and their scales)
// Every input and output lives in memory shared with the neural unit; the conversation memory is one fixed set of
// buffers (global layers: slot = position, SG slots; sliding layers: ring of 512 slots, slot = position % 512)
// that every graph reads, so nothing is copied between pieces or steps.

struct Shm {
    void* p = nullptr;
    size_t bytes = 0;
    int fd = -1;
};

// A context binary with all its graphs loaded (they share their weights).
class QnnCtx {
public:
    struct Graph {
        std::string name;
        Qnn_GraphHandle_t h = nullptr;
        std::vector<Qnn_Tensor_t> in, out;
        std::vector<std::string> inNames, outNames;
        std::vector<size_t> inBytes, outBytes;
    };
    ~QnnCtx() { release(); }
    // groupSpill > 0: share one scratch (spill-fill) area of that many bytes with the other members; the first
    // member passes groupFirst = nullptr, the rest pass the first member's context().
    // keep: when given, only the graphs it accepts are loaded (the others in the file are never deserialised and
    // reserve no working memory).
    bool load(const std::string& path, std::string* err, uint64_t groupSpill = 0, Qnn_ContextHandle_t groupFirst = nullptr,
              const std::function<bool(const std::string&)>& keep = nullptr);
    Qnn_ContextHandle_t context() const { return ctx_; }
    // Loads several files in one call with resource sharing for graphs that run one after another (what Qualcomm's
    // Genie does): scratch space and address room are shared instead of reserved per file.
    static bool loadList(const std::vector<QnnCtx*>& ctxs, const std::vector<std::string>& paths, std::string* err);
    void release();
    Graph* graph(const std::string& part);  // first graph whose name contains `part`
    std::vector<Graph>& graphs() { return graphs_; }
    // Point a tensor (input or output, by name) of graph g at shared memory.
    bool bind(Graph& g, bool input, size_t i, const Shm& m, std::string* err);
    bool exec(Graph& g, std::string* err);

private:
    bool parse(const std::string& path, std::string* err);   // map the file and read its graphs' tensors
    static void notify(Qnn_ContextHandle_t context, Qnn_GraphHandle_t graph, const char* name,
                       QnnContext_createFromBinaryAsyncNotifyType_t type, void* param, Qnn_ErrorHandle_t status);
    Qnn_ContextHandle_t ctx_ = nullptr;
    void* map_ = nullptr;
    size_t mapSize_ = 0;
    std::vector<Graph> graphs_;
    std::vector<std::unique_ptr<std::vector<uint32_t>>> dims_;
    std::vector<std::unique_ptr<std::string>> names_;
    std::map<std::pair<int, size_t>, Qnn_MemHandle_t> reg_;  // (fd, tensor bytes) -> registration in this context
};

struct GemmaStats {
    int promptWords = 0, newWords = 0, checks = 0, guessed = 0, accepted = 0;
    double msPrompt = 0, msWrite = 0, msDraft = 0, msCheck = 0, msLoad = 0;
};

class Gemma {
public:
    ~Gemma();
    bool init(const std::string& dir, const std::string& libDir, std::string* err);
    // Reads `ids` and writes up to maxNew words (greedy). drafts = guesses per check (0 = plain, one word per pass).
    std::vector<int32_t> generate(const std::vector<int32_t>& ids, int maxNew, int drafts, GemmaStats* st, std::string* err);

private:
    static constexpr int SG = 3968, SW = 512, D = 2560, NL = 42, PLE = 256, V = 262144;
    Shm alloc(size_t bytes);
    Shm& buf(const std::string& key, size_t bytes, std::string* err);
    bool bindAll(QnnCtx& c, std::string* err);
    void setWords(int T, const int32_t* ids, int n);
    void setRope(const std::string& sfx, const int* pos, int n);
    void setMasks(int T, int pos0, int real);
    void store(int T, int pos0, int count);
    bool pass(int T, std::string* err);
    int draft(int t, const _Float16* h, int pos, _Float16* hNext, std::string* err);

    QnnCtx part_[3], draft_;
    std::map<std::string, Shm> bufs_;
    std::map<int, QnnCtx::Graph*> g_[3];  // T -> graph of each piece
    QnnCtx::Graph* gd_ = nullptr;
    const uint16_t* emb_ = nullptr;        // word table, 16-bit (already in the graph's input encoding)
    const uint8_t* ple_ = nullptr;         // per-layer table, 8-bit
    size_t embBytes_ = 0, pleBytes_ = 0;
    uint16_t pleTo16_[256];                // 8-bit per-layer value -> the graph's 16-bit encoding
    float embScale_ = 0; int embOff_ = 0;
    int nStored_ = 0;                      // positions 0..nStored_-1 are in memory
    std::vector<int> ring_;                // position held by each sliding slot (-1 = empty)
    std::vector<int> own_;                 // layers with their own keys/values
};
