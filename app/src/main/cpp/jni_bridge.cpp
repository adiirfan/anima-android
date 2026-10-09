#include <jni.h>

#include <atomic>
#include <mutex>
#include <string>

#include "klein.h"
#include "pipeline.h"
#include "qimg.h"

namespace {
Pipeline g_pipe;
KleinPipeline g_klein;
KleinResult g_kleinLast;
QwenImage* g_qimg = nullptr;   // created on first use (the class maps its word table in init)
QimgResult g_qimgLast;
std::mutex g_mu;
GenResult g_last;
std::string g_error;
std::atomic<bool> g_releaseLater{false};  // asked to free memory while a picture was being made

std::string str(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r(c);
    env->ReleaseStringUTFChars(s, c);
    return r;
}

std::vector<int32_t> ints(JNIEnv* env, jintArray a) {
    std::vector<int32_t> v(env->GetArrayLength(a));
    env->GetIntArrayRegion(a, 0, (jsize)v.size(), v.data());
    return v;
}
}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_anima_npu_Engine_init(JNIEnv* env, jobject, jstring modelDir, jstring libDir) {
    std::lock_guard<std::mutex> l(g_mu);
    std::string err;
    if (!g_pipe.init(str(env, modelDir), str(env, libDir), &err)) return env->NewStringUTF(err.c_str());
    return nullptr;
}

// Returns width*height ARGB pixels, or null (see lastError). progress.onStep(step, total) is called per step.
// guide: optional width*height ARGB guide picture (null = none).
JNIEXPORT jintArray JNICALL Java_com_anima_npu_Engine_generate(JNIEnv* env, jobject, jintArray qwen, jintArray t5,
                                                                jfloatArray t5Weights, jlong seed, jint steps,
                                                                jint width, jint height, jintArray guide,
                                                                jfloat guideStrength, jfloat guideEnd,
                                                                jfloatArray noise, jobject progress) {
    std::lock_guard<std::mutex> l(g_mu);
    GenRequest req;
    req.qwenIds = ints(env, qwen);
    req.t5Ids = ints(env, t5);
    req.t5Weights.resize(env->GetArrayLength(t5Weights));
    env->GetFloatArrayRegion(t5Weights, 0, (jsize)req.t5Weights.size(), req.t5Weights.data());
    req.seed = (uint64_t)seed;
    req.steps = steps;
    req.width = width;
    req.height = height;
    if (guide) {
        std::vector<int32_t> g = ints(env, guide);
        req.guide.assign(g.begin(), g.end());
    }
    req.guideStrength = guideStrength;
    req.guideEnd = guideEnd;
    if (noise) {
        req.noise.resize(env->GetArrayLength(noise));
        env->GetFloatArrayRegion(noise, 0, (jsize)req.noise.size(), req.noise.data());
    }
    jmethodID onStep = env->GetMethodID(env->GetObjectClass(progress), "onStep", "(II)V");
    if (g_qimg) g_qimg->release();   // one model on the neural unit at a time
    g_last = g_pipe.generate(req, [&](int i, int n) { env->CallVoidMethod(progress, onStep, i, n); });
    g_error = g_last.error;
    if (g_releaseLater) { g_pipe.release(); g_releaseLater = false; }
    if (!g_error.empty()) return nullptr;
    jintArray out = env->NewIntArray((jsize)g_last.argb.size());
    env->SetIntArrayRegion(out, 0, (jsize)g_last.argb.size(), reinterpret_cast<const jint*>(g_last.argb.data()));
    return out;
}

JNIEXPORT jintArray JNICALL Java_com_anima_npu_Engine_upscale(JNIEnv* env, jobject, jintArray px, jint width, jint height) {
    std::lock_guard<std::mutex> l(g_mu);
    std::vector<int32_t> in = ints(env, px);
    std::vector<uint32_t> u(in.begin(), in.end());
    double ms = 0;
    std::vector<uint32_t> out = g_pipe.upscale(u, width, height, &g_error, &ms);
    if (out.empty()) return nullptr;
    g_error.clear();
    g_last.msDecode = ms;
    jintArray r = env->NewIntArray((jsize)out.size());
    env->SetIntArrayRegion(r, 0, (jsize)out.size(), reinterpret_cast<const jint*>(out.data()));
    return r;
}

// Picture sizes the installed model was built for ("WxH"), after init.
JNIEXPORT jobjectArray JNICALL Java_com_anima_npu_Engine_sizes(JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> l(g_mu);
    const auto& s = g_pipe.sizes();
    jobjectArray r = env->NewObjectArray((jsize)s.size(), env->FindClass("java/lang/String"), nullptr);
    for (size_t i = 0; i < s.size(); i++) env->SetObjectArrayElement(r, (jsize)i, env->NewStringUTF(s[i].c_str()));
    return r;
}

JNIEXPORT jboolean JNICALL Java_com_anima_npu_Engine_canGuide(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> l(g_mu);
    return g_pipe.canGuide();
}

// [load ms, prompt ms, drawing ms, decode ms] of the last generate (decode = upscale time after upscale()).
JNIEXPORT jdoubleArray JNICALL Java_com_anima_npu_Engine_lastTimes(JNIEnv* env, jobject) {
    double t[4] = {g_last.msLoad, g_last.msText, g_last.msDraw, g_last.msDecode};
    jdoubleArray r = env->NewDoubleArray(4);
    env->SetDoubleArrayRegion(r, 0, 4, t);
    return r;
}

JNIEXPORT jfloatArray JNICALL Java_com_anima_npu_Engine_lastLatent(JNIEnv* env, jobject) {
    jfloatArray r = env->NewFloatArray((jsize)g_last.latent.size());
    env->SetFloatArrayRegion(r, 0, (jsize)g_last.latent.size(), g_last.latent.data());
    return r;
}

JNIEXPORT jstring JNICALL Java_com_anima_npu_Engine_lastError(JNIEnv* env, jobject) {
    return env->NewStringUTF(g_error.c_str());
}

JNIEXPORT void JNICALL Java_com_anima_npu_Engine_setLoadOptions(JNIEnv*, jobject, jint bits) {
    std::lock_guard<std::mutex> l(g_mu);
    g_loadOptions = bits;
}

JNIEXPORT void JNICALL Java_com_anima_npu_Engine_setMemoryMode(JNIEnv*, jobject, jint m) {
    std::lock_guard<std::mutex> l(g_mu);
    g_pipe.setMemoryMode(m);
}

// ---- FLUX.2 klein (klein.h): its own model folder, the same neural-unit backend ----

JNIEXPORT jstring JNICALL Java_com_anima_npu_Engine_kleinInit(JNIEnv* env, jobject, jstring modelDir, jstring libDir) {
    std::lock_guard<std::mutex> l(g_mu);
    std::string err;
    if (!g_klein.init(str(env, modelDir), str(env, libDir), &err)) return env->NewStringUTF(err.c_str());
    return nullptr;
}

// width*height ARGB pixels, or null (see lastError). edit: optional width*height ARGB picture to change.
JNIEXPORT jintArray JNICALL Java_com_anima_npu_Engine_kleinGenerate(JNIEnv* env, jobject, jintArray ids, jlong seed, jint steps,
                                                                     jint width, jint height, jintArray edit, jobject progress) {
    std::lock_guard<std::mutex> l(g_mu);
    KleinRequest req;
    req.ids = ints(env, ids);
    req.seed = (uint64_t)seed;
    req.steps = steps;
    req.width = width;
    req.height = height;
    if (edit) {
        std::vector<int32_t> e = ints(env, edit);
        req.edit.assign(e.begin(), e.end());
    }
    jmethodID onStep = env->GetMethodID(env->GetObjectClass(progress), "onStep", "(II)V");
    if (g_qimg) g_qimg->release();
    g_kleinLast = g_klein.generate(req, [&](int i, int n) { env->CallVoidMethod(progress, onStep, i, n); });
    g_error = g_kleinLast.error;
    if (g_releaseLater) { g_pipe.release(); g_klein.release(); if (g_qimg) g_qimg->release(); g_releaseLater = false; }
    if (!g_error.empty()) return nullptr;
    jintArray out = env->NewIntArray((jsize)g_kleinLast.argb.size());
    env->SetIntArrayRegion(out, 0, (jsize)g_kleinLast.argb.size(), reinterpret_cast<const jint*>(g_kleinLast.argb.data()));
    return out;
}

// [load ms, prompt ms, drawing ms, decode ms] of the last Klein picture.
JNIEXPORT jdoubleArray JNICALL Java_com_anima_npu_Engine_kleinTimes(JNIEnv* env, jobject) {
    double t[4] = {g_kleinLast.msLoad, g_kleinLast.msText, g_kleinLast.msDraw, g_kleinLast.msDecode};
    jdoubleArray r = env->NewDoubleArray(4);
    env->SetDoubleArrayRegion(r, 0, 4, t);
    return r;
}

JNIEXPORT void JNICALL Java_com_anima_npu_Engine_kleinSetMemoryMode(JNIEnv*, jobject, jint m) {
    std::lock_guard<std::mutex> l(g_mu);
    g_klein.setMemoryMode(m);
}

// ---- Qwen-Image 2.1 turbo (qimg.h): its own model folder, 768x768, optional picture to edit ----

JNIEXPORT jstring JNICALL Java_com_anima_npu_Engine_qwenInit(JNIEnv* env, jobject, jstring modelDir, jstring libDir, jstring drawDir) {
    std::lock_guard<std::mutex> l(g_mu);
    // afresh every time: the app calls this again after a download, which may have switched Standard <-> Small
    delete g_qimg;
    g_qimg = new QwenImage();
    std::string err;
    if (!g_qimg->init(str(env, modelDir), str(env, libDir), &err, str(env, drawDir))) return env->NewStringUTF(err.c_str());
    return nullptr;
}

// width*height ARGB pixels, or null (see lastError). edit: optional 768x768 ARGB picture to change (the ids then carry
// its 576 picture words). progress.onStep(i, 4) as drawing step i starts (0 while reading the prompt).
JNIEXPORT jintArray JNICALL Java_com_anima_npu_Engine_qwenGenerate(JNIEnv* env, jobject, jintArray ids, jlong seed,
                                                                    jint width, jint height, jintArray edit, jobject progress) {
    std::lock_guard<std::mutex> l(g_mu);
    if (!g_qimg) { g_error = "Qwen-Image is not started"; return nullptr; }
    g_pipe.release();      // one model on the neural unit at a time
    g_klein.release();
    std::vector<uint8_t> rgb;
    if (edit) {
        std::vector<int32_t> e = ints(env, edit);
        rgb.resize(e.size() * 3);
        for (size_t i = 0; i < e.size(); i++) {
            rgb[i * 3] = (uint8_t)(e[i] >> 16); rgb[i * 3 + 1] = (uint8_t)(e[i] >> 8); rgb[i * 3 + 2] = (uint8_t)e[i];
        }
    }
    jmethodID onStep = env->GetMethodID(env->GetObjectClass(progress), "onStep", "(II)V");
    g_qimgLast = QimgResult();
    std::string err;
    bool ok = g_qimg->generate(ints(env, ids), (uint64_t)seed, width, height, &g_qimgLast, &err,
                               rgb.empty() ? nullptr : rgb.data(), [&](int i, int n) { env->CallVoidMethod(progress, onStep, i, n); });
    g_error = ok ? "" : err;
    if (g_releaseLater) { g_pipe.release(); g_klein.release(); g_qimg->release(); g_releaseLater = false; }
    if (!ok) return nullptr;
    jintArray out = env->NewIntArray((jsize)g_qimgLast.argb.size());
    env->SetIntArrayRegion(out, 0, (jsize)g_qimgLast.argb.size(), reinterpret_cast<const jint*>(g_qimgLast.argb.data()));
    return out;
}

// [picture ms, prompt (reader) ms, drawing ms (prompt pass + steps), decode ms] of the last Qwen-Image picture.
JNIEXPORT jdoubleArray JNICALL Java_com_anima_npu_Engine_qwenTimes(JNIEnv* env, jobject) {
    double t[4] = {g_qimgLast.msVision, g_qimgLast.msReader, g_qimgLast.msPrefix + g_qimgLast.msSteps, g_qimgLast.msDecode};
    jdoubleArray r = env->NewDoubleArray(4);
    env->SetDoubleArrayRegion(r, 0, 4, t);
    return r;
}

// Total time spent running graphs on the neural unit so far (ns); the usage monitor turns it into a busy share.
// No lock: it must not wait for a running picture.
JNIEXPORT jlong JNICALL Java_com_anima_npu_Engine_npuBusyNs(JNIEnv*, jobject) {
    return (jlong)g_npuBusyNs.load();
}

// Called from the screen thread when the app leaves the screen: never wait for a running picture.
JNIEXPORT void JNICALL Java_com_anima_npu_Engine_release(JNIEnv*, jobject) {
    std::unique_lock<std::mutex> l(g_mu, std::try_to_lock);
    if (!l.owns_lock()) { g_releaseLater = true; return; }
    g_pipe.release();
    g_klein.release();
    if (g_qimg) g_qimg->release();
}

}  // extern "C"
