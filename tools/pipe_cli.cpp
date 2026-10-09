// Phone-side test of the whole picture pipeline without the app.
// usage: pipe_cli <modeldir> <libdir> <WxH> <steps> <seed> <qwen.i32> <t5.i32> [guide.argb strength end] [memmode]
// guide.argb: W*H uint32 ARGB. Writes out.argb (W*H uint32), out.lat (float32) and prints timings and peak memory.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "pipeline.h"

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
    if (argc < 8) { fprintf(stderr, "usage: pipe_cli <modeldir> <libdir> <WxH> <steps> <seed> <qwen.i32> <t5.i32> [guide.argb strength end] [memmode]\n"); return 2; }
    Pipeline p;
    std::string err;
    if (!p.init(argv[1], argv[2], &err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    printf("sizes:");
    for (auto& s : p.sizes()) printf(" %s", s.c_str());
    printf(" | guide %s\n", p.canGuide() ? "yes" : "no");
    GenRequest r;
    sscanf(argv[3], "%dx%d", &r.width, &r.height);
    r.steps = atoi(argv[4]);
    r.seed = strtoull(argv[5], nullptr, 10);
    r.qwenIds = readAll<int32_t>(argv[6]);
    r.t5Ids = readAll<int32_t>(argv[7]);
    int mm = 1;
    if (argc >= 11) {
        r.guide = readAll<uint32_t>(argv[8]);
        r.guideStrength = atof(argv[9]);
        r.guideEnd = atof(argv[10]);
        if (argc >= 12) mm = atoi(argv[11]);
    } else if (argc == 9) {
        mm = atoi(argv[8]);
    }
    p.setMemoryMode(mm);
    GenResult g = p.generate(r, [](int i, int n) { fprintf(stderr, "\rstep %d/%d", i, n); });
    fprintf(stderr, "\n");
    if (!g.error.empty()) { fprintf(stderr, "generate: %s\n", g.error.c_str()); return 1; }
    printf("%dx%d %d steps: prompt+guide %.0f ms, loading %.0f ms, drawing %.0f ms, decode %.0f ms | peak memory %ld MB\n",
           r.width, r.height, r.steps, g.msText, g.msLoad, g.msDraw, g.msDecode, memKb("VmHWM:") / 1024);
    std::ofstream("out.argb", std::ios::binary).write(reinterpret_cast<const char*>(g.argb.data()), g.argb.size() * 4);
    std::ofstream("out.lat", std::ios::binary).write(reinterpret_cast<const char*>(g.latent.data()), g.latent.size() * 4);
    if (getenv("UPSCALE")) {
        double ms = 0;
        std::vector<uint32_t> up = p.upscale(g.argb, r.width, r.height, &err, &ms);
        if (up.empty()) { fprintf(stderr, "upscale: %s\n", err.c_str()); return 1; }
        printf("upscale %dx%d in %.0f ms | peak memory %ld MB\n", 4 * r.width, 4 * r.height, ms, memKb("VmHWM:") / 1024);
        std::ofstream("up.argb", std::ios::binary).write(reinterpret_cast<const char*>(up.data()), up.size() * 4);
    }
    return 0;
}
