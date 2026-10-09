// usage: qimg_cli <modeldir> <libdir> <ids.i32> <seed> <WxH> [edit.rgb]   -> qimg_out.argb
// edit.rgb: the 768x768 picture to edit as raw RGB bytes (the ids then carry its 576 <|image_pad|> words)
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <vector>

#include "qimg.h"

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: qimg_cli <modeldir> <libdir> <ids.i32> <seed> <WxH>\n"); return 2; }
    std::ifstream f(argv[3], std::ios::binary | std::ios::ate);
    std::vector<int32_t> ids((size_t)f.tellg() / 4);
    f.seekg(0);
    f.read((char*)ids.data(), ids.size() * 4);
    int W = 768, H = 768;
    sscanf(argv[5], "%dx%d", &W, &H);
    std::vector<uint8_t> rgb;
    if (argc > 6) {
        std::ifstream e(argv[6], std::ios::binary | std::ios::ate);
        rgb.resize((size_t)e.tellg());
        e.seekg(0);
        e.read((char*)rgb.data(), rgb.size());
        if (rgb.size() != (size_t)768 * 768 * 3) { fprintf(stderr, "edit picture must be 768x768 raw RGB\n"); return 2; }
    }
    std::string err;
    QwenImage q;
    if (!q.init(argv[1], argv[2], &err)) { fprintf(stderr, "init: %s\n", err.c_str()); _exit(1); }
    QimgResult r;
    if (!q.generate(ids, strtoull(argv[4], nullptr, 10), W, H, &r, &err, rgb.empty() ? nullptr : rgb.data())) { fprintf(stderr, "generate: %s\n", err.c_str()); _exit(1); }
    std::ofstream("qimg_out.argb", std::ios::binary).write((const char*)r.argb.data(), r.argb.size() * 4);
    if (r.msVision > 0) printf("picture (vision + encoder, incl. loading) %.0f ms\n", r.msVision);
    printf("reader %.0f ms (incl. loading), drawing model load %.0f ms, prompt pass %.0f ms, 4 steps %.0f ms, decode %.0f ms\n",
           r.msReader, r.msLoad, r.msPrefix, r.msSteps, r.msDecode);
    if (r.msSwapLoad > 0) printf("  of the 4 steps, loading/releasing files took %.0f ms\n", r.msSwapLoad);
    fflush(stdout);
    _exit(0);
}
