#!/bin/bash
# Builds qnn_cli for the phone from the app's engine sources.
set -e
NDK=${NDK:-$HOME/Android/Sdk/ndk/28.2.13676358}
QNN=${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the QAIRT 2.50.40 folder}
D=$(cd "$(dirname "$0")/.." && pwd)
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++ -std=c++17 -O2 -Wno-unused-parameter \
  -I$QNN/include/QNN -I$D/app/src/main/cpp $D/tools/qnn_cli.cpp $D/app/src/main/cpp/qnn_runner.cpp \
  -o $D/tools/qnn_cli -llog -ldl -static-libstdc++
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++ -std=c++17 -O2 -Wno-unused-parameter \
  -I$QNN/include/QNN -I$D/app/src/main/cpp $D/tools/pipe_cli.cpp $D/app/src/main/cpp/qnn_runner.cpp $D/app/src/main/cpp/pipeline.cpp \
  -o $D/tools/pipe_cli -llog -ldl -static-libstdc++
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++ -std=c++17 -O2 -Wno-unused-parameter \
  -I$QNN/include/QNN -I$D/app/src/main/cpp $D/tools/klein_cli.cpp $D/app/src/main/cpp/qnn_runner.cpp $D/app/src/main/cpp/klein.cpp \
  -o $D/tools/klein_cli -llog -ldl -static-libstdc++
echo built $D/tools/qnn_cli $D/tools/pipe_cli $D/tools/klein_cli
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++ -std=c++17 -O2 -Wno-unused-parameter \
  -I$QNN/include/QNN -I$D/app/src/main/cpp $D/tools/gemma_cli.cpp $D/app/src/main/cpp/qnn_runner.cpp $D/app/src/main/cpp/gemma.cpp \
  -o $D/tools/gemma_cli -llog -ldl -static-libstdc++
echo built $D/tools/gemma_cli
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++ -std=c++17 -O2 -Wno-unused-parameter \
  -I$QNN/include/QNN -I$D/app/src/main/cpp $D/tools/qimg_cli.cpp $D/app/src/main/cpp/qnn_runner.cpp $D/app/src/main/cpp/gemma.cpp $D/app/src/main/cpp/qimg.cpp \
  -o $D/tools/qimg_cli -llog -ldl -static-libstdc++
echo built $D/tools/qimg_cli
