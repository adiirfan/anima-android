// usage: loadtest <libdir> <piece.bin> <max copies> [group 0|1]: loads copies until the phone refuses, prints the count
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unistd.h>
#include <vector>
#include "gemma.h"
int main(int argc, char** argv) {
    std::string err;
    if (!QnnBackend::get().open(argv[1], &err)) { fprintf(stderr, "%s\n", err.c_str()); _exit(1); }
    int maxc = atoi(argv[3]);
    bool group = argc > 4 && atoi(argv[4]);
    uint64_t spill = group ? QnnGraph::spillSize(argv[2]) : 0;
    std::vector<std::unique_ptr<QnnCtx>> v;
    for (int i = 0; i < maxc; i++) {
        v.emplace_back(new QnnCtx());
        if (!v.back()->load(argv[2], &err, spill, i && group ? v[0]->context() : nullptr)) {
            printf("loaded %d copies, then: %s\n", i, err.c_str()); fflush(stdout); _exit(0);
        }
    }
    printf("loaded all %d copies (spill %.0f MB each)\n", maxc, spill / 1e6); fflush(stdout);
    _exit(0);
}
