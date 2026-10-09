// Phone-side test of the app's engine code without the app: load one graph of a compiled piece, run it,
// report load time, run time and memory. Built with tools/build_cli.sh, run over adb.
// usage: qnn_cli <libdir> <piece.bin> <graph name or -> <runs> [input.raw ...]   (inputs float32, graph order)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "qnn_runner.h"

static long memKb(const char* file, const char* key) {
    std::ifstream f(file);
    std::string k;
    long v = -1;
    while (f >> k) {
        if (k == key) { f >> v; return v; }
    }
    return v;
}
static double ms(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: qnn_cli <libdir> <piece.bin> <graph|-> <runs> [in.raw ...]\n"); return 2; }
    std::string err, graph = argv[3] == std::string("-") ? "" : argv[3];
    int runs = atoi(argv[4]);
    if (!QnnBackend::get().open(argv[1], &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    long avail0 = memKb("/proc/meminfo", "MemAvailable:"), pss0 = memKb("/proc/self/smaps_rollup", "Pss:");
    QnnGraph g;
    auto t = std::chrono::steady_clock::now();
    uint64_t spill = QnnGraph::spillSize(argv[2], graph);
    if (!g.load(argv[2], &err, false, nullptr, 0, graph)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    double tl = ms(t);
    long avail1 = memKb("/proc/meminfo", "MemAvailable:"), pss1 = memKb("/proc/self/smaps_rollup", "Pss:");
    printf("loaded %s graph %s in %.0f ms, scratch %.0f MB\n%s", argv[2], graph.empty() ? "(first)" : graph.c_str(), tl, spill / 1e6,
           g.describe().c_str());
    std::vector<std::vector<float>> in(g.inputs().size()), out(g.outputs().size());
    for (size_t i = 0; i < in.size(); i++) {
        in[i].assign(g.inputs()[i].count, 0.f);
        if ((int)i + 5 < argc) {
            std::ifstream f(argv[i + 5], std::ios::binary);
            f.read(reinterpret_cast<char*>(in[i].data()), in[i].size() * 4);
            if (!f) { fprintf(stderr, "short input %s\n", argv[i + 5]); return 1; }
        }
    }
    for (size_t i = 0; i < out.size(); i++) out[i].resize(g.outputs()[i].count);
    std::vector<const float*> ip; std::vector<float*> op;
    for (auto& v : in) ip.push_back(v.data());
    for (auto& v : out) op.push_back(v.data());
    double best = 1e9, sum = 0;
    for (int r = 0; r < runs; r++) {
        t = std::chrono::steady_clock::now();
        if (!g.run(ip, op, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        double d = ms(t);
        best = std::min(best, d); sum += d;
    }
    long avail2 = memKb("/proc/meminfo", "MemAvailable:"), pss2 = memKb("/proc/self/smaps_rollup", "Pss:");
    printf("run: best %.1f ms, mean %.1f ms over %d\n", best, sum / runs, runs);
    printf("memory: process %ld -> %ld -> %ld MB, phone free memory dropped %ld MB by load, %ld MB by end\n",
           pss0 / 1024, pss1 / 1024, pss2 / 1024, (avail0 - avail1) / 1024, (avail0 - avail2) / 1024);
    for (size_t i = 0; i < out.size(); i++) {
        std::ofstream f("out_" + std::to_string(i) + ".raw", std::ios::binary);
        f.write(reinterpret_cast<const char*>(out[i].data()), out[i].size() * 4);
    }
    return 0;
}
