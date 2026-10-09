// Phone-side test of the Klein pipeline without the app.
// usage: klein_cli <modeldir> <libdir> <WxH> <steps> <seed> <ids.i32> [edit.argb] [memmode]
// Writes out.argb (W*H uint32) and prints timings and peak memory.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <string>
#include <vector>
#include "klein.h"

template <class T> static std::vector<T> readAll(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::vector<T> v(f ? (size_t)f.tellg() / sizeof(T) : 0);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    return v;
}
static long memKb(const char* key) {
    std::ifstream f("/proc/self/status");
    std::string k;
    long v = -1;
    while (f >> k) if (k == key) { f >> v; break; }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: klein_cli <modeldir> <libdir> <WxH> <steps> <seed> <ids.i32> [edit.argb] [memmode]\n"); return 2; }
    KleinPipeline p;
    std::string err;
    if (!p.init(argv[1], argv[2], &err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    KleinRequest r;
    sscanf(argv[3], "%dx%d", &r.width, &r.height);
    r.steps = atoi(argv[4]);
    r.seed = strtoull(argv[5], nullptr, 10);
    r.ids = readAll<int32_t>(argv[6]);
    int mm = 1;
    if (argc >= 8 && std::string(argv[7]) != "-") r.edit = readAll<uint32_t>(argv[7]);
    if (argc >= 9) mm = atoi(argv[8]);
    p.setMemoryMode(mm);
    KleinResult g = p.generate(r, [](int i, int n) { fprintf(stderr, "\rstep %d/%d", i, n); });
    fprintf(stderr, "\n");
    if (!g.error.empty()) { fprintf(stderr, "generate: %s\n", g.error.c_str()); fflush(stdout); _exit(1); }
    printf("%dx%d %s %d steps: prompt %.0f ms, loading %.0f ms, drawing %.0f ms, decode %.0f ms | peak memory %ld MB\n",
           r.width, r.height, r.edit.empty() ? "picture" : "edit", r.steps, g.msText, g.msLoad, g.msDraw, g.msDecode,
           memKb("VmHWM:") / 1024);
    std::ofstream("out.argb", std::ios::binary).write(reinterpret_cast<const char*>(g.argb.data()), g.argb.size() * 4);
    fflush(stdout);
    _exit(0);  // skip global teardown (QNN's own exit-time clean-up can crash after a failed load)
}
