// usage: gemma_cli <modeldir> <libdir> <ids.i32> <max new words> <guesses per check (0 = plain)>
// Writes the new word ids to gemma_out.i32 and prints the timings.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <vector>

#include "gemma.h"
#include <map>
extern std::map<std::string, std::pair<double, int>> g_execMs;

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: gemma_cli <modeldir> <libdir> <ids.i32> <max new> <guesses>\n"); return 2; }
    std::ifstream f(argv[3], std::ios::binary | std::ios::ate);
    std::vector<int32_t> ids((size_t)f.tellg() / 4);
    f.seekg(0);
    f.read((char*)ids.data(), ids.size() * 4);
    std::string err;
    Gemma g;
    if (!g.init(argv[1], argv[2], &err)) { fprintf(stderr, "init: %s\n", err.c_str()); _exit(1); }
    GemmaStats st;
    auto out = g.generate(ids, atoi(argv[4]), atoi(argv[5]), &st, &err);
    if (!err.empty()) { fprintf(stderr, "generate: %s\n", err.c_str()); _exit(1); }
    std::ofstream("gemma_out.i32", std::ios::binary).write((const char*)out.data(), out.size() * 4);
    printf("prompt %d words in %.0f ms (%.0f words/s)\n", st.promptWords, st.msPrompt, st.promptWords * 1000.0 / st.msPrompt);
    printf("wrote %d words in %.0f ms (%.1f words/s); checks %d, guesses accepted %d/%d; drafting %.0f ms, checking %.0f ms\n",
           st.newWords, st.msWrite, (st.newWords - 1) * 1000.0 / st.msWrite, st.checks, st.accepted, st.guessed, st.msDraft, st.msCheck);
    for (auto& e : g_execMs) printf("  %-32s %5d runs, %.1f ms each\n", e.first.c_str(), e.second.second, e.second.first / e.second.second);
    fflush(stdout);
    _exit(0);
}
