# Anima on the neural unit

Android app that draws pictures entirely on the Snapdragon 8 Elite neural unit (Qualcomm QNN / HTP v79), with three
models to choose from:

- **Anima Turbo**: anime pictures in six sizes, about 6–7 s each at 8 steps on an iQOO 13, optional guide picture and
  4× upscaler.
- **FLUX.2 [klein] 4B**: pictures from plain sentences and picture editing.
- **Qwen-Image 2.1** (4-step turbo): 768×768 pictures in about 1½ minutes, picture editing in about 3; a Small
  version with a 2B prompt reader is a smaller download.

Pictures go into a **queue** that keeps drawing in the background, with a progress notification and a "Picture ready"
notification for each one. A **Chat** tab talks to [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
running on the same neural unit (through llama.cpp's Hexagon backend); ask it for a picture and it picks Anima or
Qwen-Image, writes the prompt and adds it to the queue, and the picture appears in the conversation. The app also
**updates itself** from this repo's releases.

The app downloads the model files from Hugging Face:
[anima-qnn-8elite](https://huggingface.co/sebekblessing/anima-qnn-8elite),
[klein-qnn-8elite](https://huggingface.co/sebekblessing/klein-qnn-8elite),
[image21-qnn-8elite](https://huggingface.co/sebekblessing/image21-qnn-8elite), and Ling from the `ling/` folder of
anima-qnn-8elite. Anima and Qwen-Image are licensed for **non-commercial use only**; see the licence and notice files
in each repo. Ling-3.0-tiny is MIT-licensed by inclusionAI.

To build those files yourself from the original models, each model repo has the scripts and a step-by-step guide in
its `build/` folder.

## Disclaimer

All of the code in this repository was written by an AI coding assistant (Claude, by Anthropic). It is shared
as-is, without any warranty, and you use it at your own risk. I am not responsible for any problem it causes,
including crashes, lost data, wrong results, or a phone that runs hot or is damaged. The model files have their
own licences; see their Hugging Face pages.

## Build

Needs the Android SDK/NDK r28 and Qualcomm QAIRT 2.50.40 (for the QNN headers): put `qnn.sdk.dir=<QAIRT folder>` in
`local.properties`, or set `QNN_SDK_ROOT`.

    ./gradlew assembleDebug

The QNN runtime libraries for the phone are in `app/src/main/jniLibs/arm64-v8a` (from QAIRT 2.50.40).

The chat runs llama.cpp's server as a separate program inside the app. Build llama.cpp for Android with
`-DGGML_HEXAGON=ON` (Hexagon SDK 6.6), then copy it in, stripped:

    LLAMA_BUILD=<llama.cpp build folder> tools/pack-llama.sh

To publish an app update (builds with a time stamp as the build number and publishes `anima.apk` + `latest.json` as
a new release of this repo, where installed copies look for updates; needs the GitHub CLI, logged in):

    tools/publish-app.sh "what changed"

## Layout

- `app/src/main/cpp/` — engine on the QNN C API: `qnn_runner` (load/run one compiled piece), `pipeline` (Anima),
  `klein` (FLUX.2 klein), `qimg` (Qwen-Image), `gemma` (shared context code), `jni_bridge`.
- `app/src/main/java/com/anima/npu/` — Compose screens, tokenizers (ComfyUI's rules), downloads, gallery, usage
  bar, self-test; the queue (`QueueStore`, `QueueService`, `Drawer`), the chat (`LocalLing` starts the server,
  `ChatController` runs a turn and the `draw_picture` tool, `ChatScreen`), `NpuGate` (the queue and the chat take
  turns on the neural unit) and `AppUpdate`.
- `tools/*_cli.cpp` — command-line test programs that run a model on a phone over adb (`tools/build_cli.sh`).
