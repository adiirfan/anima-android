# Anima on the neural unit

Android app that draws pictures and short clips entirely on the phone's neural unit (Qualcomm QNN / HTP), on the
**Snapdragon 8 Elite** (HTP v79) and the **Snapdragon 8 Elite Gen 5** (HTP v81). Other phones get a "Device not
supported" page. Pick a model from one list (times measured on an iQOO 13 and a Samsung S25 Edge, both 8 Elite):

- **Anima Turbo**: anime pictures in six sizes, about 6–17 s each, optional guide picture and 4× upscaler.
- **FLUX.2 [klein] 4B**: pictures from plain sentences in about 8 s, picture editing in about 20 s.
- **Qwen-Image 2.1 Turbo** (8 steps): 768×768 pictures in about 2½ minutes, picture editing in about 5.
- **Fibo 2.0**: picture editing only, about 75 s.
- **MOSS-Image** (8-step version): pictures in about 1¾ minutes.
- **Wan 2.2 5B**: turns a picture into a short moving clip (768×768, 96 frames) in about 2½ minutes, saved as MP4.

Pictures go into a **queue** that keeps drawing in the background, with a progress notification and a "Picture ready"
notification for each one. A **Chat** tab talks to [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
running on the same neural unit (through llama.cpp's Hexagon backend); ask it for a picture and it picks Anima or
Qwen-Image, writes the prompt and adds it to the queue, and the picture appears in the conversation. The app also
**updates itself** from this repo's releases.

On first launch a setup screen lists the models this phone can run, with their download sizes, as tick boxes; the
chosen ones download at the same time, and Settings opens the same screen later. The compiled model files only run on
the chip they were built for, so the app picks the repos by chip. On the Snapdragon 8 Elite it downloads from:
[anima-qnn-8elite](https://huggingface.co/sebekblessing/anima-qnn-8elite),
[klein-4b-qnn-8elite](https://huggingface.co/sebekblessing/klein-4b-qnn-8elite),
[qwen-image-2.1-qnn-8elite](https://huggingface.co/sebekblessing/qwen-image-2.1-qnn-8elite),
[fibo-qnn-8elite](https://huggingface.co/sebekblessing/fibo-qnn-8elite),
[moss-qnn-8elite](https://huggingface.co/sebekblessing/moss-qnn-8elite),
[wan22-qnn-8elite](https://huggingface.co/sebekblessing/wan22-qnn-8elite).
On the 8 Elite Gen 5 it uses
[anima-qnn-8elite-gen5](https://huggingface.co/sebekblessing/anima-qnn-8elite-gen5),
[fibo-qnn-8elite-gen5](https://huggingface.co/sebekblessing/fibo-qnn-8elite-gen5),
[moss-qnn-8elite-gen5](https://huggingface.co/sebekblessing/moss-qnn-8elite-gen5),
[wan22-qnn-8elite-gen5](https://huggingface.co/sebekblessing/wan22-qnn-8elite-gen5) and
[klein-4b-qnn-8elite-gen5](https://huggingface.co/sebekblessing/klein-4b-qnn-8elite-gen5) (768x768, square only);
Qwen-Image is not built for that chip's neural unit yet.

Every picture model also has a build for the phone's graphics chip, which the Models screen offers next to the
neural-unit one ("Draws on"): [anima-gpu](https://huggingface.co/sebekblessing/anima-gpu),
[klein-4b-gpu](https://huggingface.co/sebekblessing/klein-4b-gpu),
[qwen-image-2.1-gpu](https://huggingface.co/sebekblessing/qwen-image-2.1-gpu) (pictures only, no edits),
[fibo-gpu](https://huggingface.co/sebekblessing/fibo-gpu), [moss-gpu](https://huggingface.co/sebekblessing/moss-gpu)
and [wan22-gpu](https://huggingface.co/sebekblessing/wan22-gpu). A build that is not available for a phone is shown
greyed out, not hidden. The two chat models (Ling-3.0-tiny and the smaller Qwen3.5 0.8B) are plain llama.cpp
files that run on both chips, from [anima-chat-models](https://huggingface.co/sebekblessing/anima-chat-models). Anima, Qwen-Image and Fibo are licensed for **non-commercial use only**; FLUX.2 [klein] 4B,
MOSS-Image and Wan 2.2 are Apache 2.0 (the Wan Turbo version's page names no licence). See the licence and notice
files in each repo. Ling-3.0-tiny is MIT-licensed by inclusionAI.

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
  `klein` (FLUX.2 klein), `qimg` (Qwen-Image), `fibo`, `moss`, `wan`, `gemma` (shared context code), `jni_bridge`.
- `app/src/main/java/com/anima/npu/` — Compose screens, tokenizers (ComfyUI's rules), downloads, gallery, usage
  bar, self-test; the queue (`QueueStore`, `QueueService`, `Drawer`), the chat (`LocalLing` starts the server,
  `ChatController` runs a turn and the `draw_picture` tool, `ChatScreen`), `NpuGate` (the queue and the chat take
  turns on the neural unit) and `AppUpdate`.
- `tools/*_cli.cpp` — command-line test programs that run a model on a phone over adb (`tools/build_cli.sh`).
