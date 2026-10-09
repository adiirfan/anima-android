#!/bin/bash
# Copy llama.cpp's phone build (with our chip patches, see ~/llamacpp-hexagon-ling3.patch) into the app, stripped.
# The server is renamed libllama-server.so so Android unpacks it next to the libraries: the only place an app may
# run a program from, and the place the AI chip loads its own half (libggml-htp-v79.so) from.
set -e
cd "$(dirname "$0")/.."
B=${LLAMA_BUILD:-$HOME/llamacpp-hexagon/build-snapdragon}
OUT=app/src/main/jniLibs/arm64-v8a
STRIP=${ANDROID_NDK_ROOT:-$HOME/android-ndk-r28c}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
cp $B/bin/llama-server $OUT/libllama-server.so
for f in libllama-server-impl libllama-common libmtmd libllama libggml libggml-base libggml-cpu libggml-opencl libggml-hexagon; do
  cp $B/bin/$f.so $OUT/
done
for f in $OUT/libllama*.so $OUT/libmtmd.so $OUT/libggml*.so; do $STRIP --strip-unneeded "$f"; done
# the chip's half is built for the DSP, which the ARM strip does not understand
cp $B/ggml/src/ggml-hexagon/libggml-htp-v79.so $OUT/libggml-htp-v79.so
ls -la $OUT | grep -E "llama|ggml|mtmd"
du -ch $OUT/libllama*.so $OUT/libmtmd.so $OUT/libggml*.so | tail -1
