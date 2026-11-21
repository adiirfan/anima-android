// Tiny picture decoders (TAESD family) for model_tae.h. Activations: fp16, rows of pixels, channels innermost
// (H, W, C), C a multiple of 4. Filters: fp16 [out][ky][kx][in]. Sums in fp32.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Y = relu?(conv_k(X') + B + R), X' = X doubled in size first when up = 1 (nearest), padding k / 2;
// up = -1: stride 2 (the input is twice the output's size, as the encoders' shrinking filters); up = -2: stride 2
// padded only right / bottom (Wan's VAE: output pixel y reads input rows 2y .. 2y + K - 1).
// One work item: 4 output channels of PX pixels side by side in one row.
#ifndef PX
#define PX 4
#endif
__kernel void conv(__global const half *X, __global const half *Wt, __global const float *B, __global const half *R,
                   __global half *Y, int H, int W, int Ci, int Co, int K, int up, int relu, int hasB, int hasR) {
    const int c4 = get_global_id(0), x0 = get_global_id(1) * PX, y = get_global_id(2);
    if (c4 * 4 >= Co || x0 >= W || y >= H) return;
    const int dn = up < 0, u = up > 0, p = up == -2 ? 0 : K / 2, Wi = dn ? 2 * W : W >> u, Hin = dn ? 2 * H : H, Win = dn ? 2 * W : W;
    float4 acc[PX];
    const float4 b = hasB ? vload4(c4, B) : (float4)(0.0f);
    for (int j = 0; j < PX; j++) acc[j] = b;
    const int KK = K * K * Ci;
    __global const half *w0 = Wt + (size_t)(c4 * 4) * KK;
    for (int ky = 0; ky < K; ky++) {
        const int iy = (dn ? 2 * y : y) + ky - p;
        if (iy < 0 || iy >= Hin) continue;
        __global const half *xr = X + (size_t)(iy >> u) * Wi * Ci;
        for (int kx = 0; kx < K; kx++) {
            __global const half *wk = w0 + (ky * K + kx) * Ci;
            int ix[PX]; bool ok[PX];
            for (int j = 0; j < PX; j++) { const int xx = (dn ? 2 * (x0 + j) : x0 + j) + kx - p; ok[j] = xx >= 0 && xx < Win; ix[j] = (ok[j] ? xx : 0) >> u; }
            for (int ci = 0; ci < Ci; ci += 4) {
                const float4 wa = vload_half4(0, wk + ci), wb = vload_half4(0, wk + KK + ci);
                const float4 wc = vload_half4(0, wk + 2 * KK + ci), wd = vload_half4(0, wk + 3 * KK + ci);
                for (int j = 0; j < PX; j++) {
                    const float4 xv = ok[j] ? vload_half4(0, xr + (size_t)ix[j] * Ci + ci) : (float4)(0.0f);
                    acc[j] += (float4)(dot(xv, wa), dot(xv, wb), dot(xv, wc), dot(xv, wd));
                }
            }
        }
    }
    for (int j = 0; j < PX; j++) {
        if (x0 + j >= W) break;
        const size_t o = ((size_t)y * W + x0 + j) * Co + c4 * 4;
        float4 a = acc[j];
        if (hasR) a += vload_half4(0, R + o);
        if (relu) a = fmax(a, 0.0f);
        vstore_half4(a, 0, Y + o);
    }
}

// x = 3 tanh(x / 3), in place
__kernel void clampt(__global half *X, int n) {
    const int i = get_global_id(0);
    if (i < n) vstore_half(3.0f * tanh(vload_half(i, X) / 3.0f), i, X);
}

// Picture-memory version (Adreno reads these through the texture cache): activations are image2d RGBA half,
// texel (x * C4 + c4, y) = channels 4 c4 .. 4 c4 + 3 of pixel (x, y). Filters: image2d, texel (tap * Ci4 + ci4, o) =
// input channels 4 ci4.. of output o at tap (ky, kx). One work item: OC output channels (OC/4 groups) of PX pixels.
// ACC: float or half sums.
#ifndef OC
#define OC 8
#endif
#ifndef ACC
#define ACC float
#endif
#define ACC4 CAT(ACC, 4)
#define CAT(a, b) CAT_(a, b)
#define CAT_(a, b) a##b
__constant sampler_t smpi = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__kernel void convi(__read_only image2d_t X, __read_only image2d_t Wt, __global const float *B, __read_only image2d_t R,
                    __write_only image2d_t Y, int H, int W, int Ci, int Co, int K, int up, int relu, int hasB, int hasR) {
    const int og = get_global_id(0), x0 = get_global_id(1) * PX, y = get_global_id(2);
#define G (OC / 4)
    if (og * OC >= Co || x0 >= W || y >= H) return;
    const int p = up == -2 ? 0 : K / 2, Ci4 = Ci / 4, Co4 = Co / 4, dn = up < 0, u = up > 0, Hin = dn ? 2 * H : H, Win = dn ? 2 * W : W;
    ACC4 acc[G][PX];
    for (int g = 0; g < G; g++) {
        const float4 b = hasB && og * G + g < Co4 ? vload4(og * G + g, B) : (float4)(0.0f);
        for (int j = 0; j < PX; j++) acc[g][j] = CAT(convert_, ACC4)(b);
    }
    for (int ky = 0; ky < K; ky++) {
        const int iy = (dn ? 2 * y : y) + ky - p;
        if (iy < 0 || iy >= Hin) continue;
        const int sy = iy >> u;
        for (int kx = 0; kx < K; kx++) {
            int ix[PX]; bool ok[PX];
            for (int j = 0; j < PX; j++) { const int xx = (dn ? 2 * (x0 + j) : x0 + j) + kx - p; ok[j] = xx >= 0 && xx < Win; ix[j] = ((ok[j] ? xx : 0) >> u) * Ci4; }
            const int tb = (ky * K + kx) * Ci4;
            for (int c = 0; c < Ci4; c++) {
                half4 xv[PX];
                for (int j = 0; j < PX; j++) xv[j] = ok[j] ? read_imageh(X, smpi, (int2)(ix[j] + c, sy)) : (half4)(0.0h);
                for (int g = 0; g < G; g++) {
                    const int o = (og * G + g) * 4;
                    const half4 wa = read_imageh(Wt, smpi, (int2)(tb + c, o)), wb = read_imageh(Wt, smpi, (int2)(tb + c, o + 1));
                    const half4 wc = read_imageh(Wt, smpi, (int2)(tb + c, o + 2)), wd = read_imageh(Wt, smpi, (int2)(tb + c, o + 3));
                    for (int j = 0; j < PX; j++) {
                        const ACC4 xa = CAT(convert_, ACC4)(xv[j]);
                        acc[g][j] += (ACC4)(dot(xa, CAT(convert_, ACC4)(wa)), dot(xa, CAT(convert_, ACC4)(wb)),
                                            dot(xa, CAT(convert_, ACC4)(wc)), dot(xa, CAT(convert_, ACC4)(wd)));
                    }
                }
            }
        }
    }
    for (int g = 0; g < G; g++) {
        const int c4 = og * G + g;
        if (c4 >= Co4) break;
        for (int j = 0; j < PX; j++) {
            const int xx = x0 + j;
            if (xx >= W) break;
            float4 a = convert_float4(acc[g][j]);
            if (hasR) a += convert_float4(read_imageh(R, smpi, (int2)(xx * Co4 + c4, y)));
            if (relu) a = fmax(a, 0.0f);
            write_imageh(Y, (int2)(xx * Co4 + c4, y), convert_half4(a));
        }
    }
}
// x = 3 tanh(x / 3) on a picture-memory activation, in place (w texels by h rows)
__kernel void clampi(__read_only image2d_t X, __write_only image2d_t Y, int w, int h) {
    const int i = get_global_id(0), y = get_global_id(1);
    if (i >= w || y >= h) return;
    const float4 v = convert_float4(read_imageh(X, smpi, (int2)(i, y)));
    write_imageh(Y, (int2)(i, y), convert_half4(3.0f * tanh(v / 3.0f)));
}

// GroupNorm statistics of a picture-memory activation (w texels x h rows, C4 texels per pixel, groups of GC4 texels):
// one work group (256 items) per group -> st[2 g] = mean, st[2 g + 1] = 1 / sqrt(var + eps)
__kernel void gnstats(__read_only image2d_t X, __global float *st, int Wd, int H, int C4, int GC4) {
    const int g = get_group_id(0), l = get_local_id(0);
    __local float s1[256], s2[256];
    float a = 0.0f, b = 0.0f;
    const int per = Wd * H * GC4;
    for (int i = l; i < per; i += 256) {
        const int c = i % GC4, px = i / GC4, x = px % Wd, y = px / Wd;
        const float4 v = convert_float4(read_imageh(X, smpi, (int2)(x * C4 + g * GC4 + c, y)));
        a += v.x + v.y + v.z + v.w; b += dot(v, v);
    }
    s1[l] = a; s2[l] = b;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 128; k > 0; k >>= 1) { if (l < k) { s1[l] += s1[l + k]; s2[l] += s2[l + k]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0) {
        const float n = 4.0f * per, m = s1[0] / n;
        st[2 * g] = m; st[2 * g + 1] = rsqrt(fmax(s2[0] / n - m * m, 0.0f) + 1e-5f);
    }
}
// relu((x - mean) * rstd * gamma + beta), per texel
__kernel void gnapply(__read_only image2d_t X, __write_only image2d_t Y, __global const float *st, __global const float *gw,
                      __global const float *gb, int Wd, int H, int C4, int GC4) {
    const int i = get_global_id(0), y = get_global_id(1);
    if (i >= Wd * C4 || y >= H) return;
    const int c4 = i % C4, g = c4 / GC4;
    const float4 v = convert_float4(read_imageh(X, smpi, (int2)(i, y)));
    const float4 r = fmax((v - st[2 * g]) * st[2 * g + 1] * vload4(c4, gw) + vload4(c4, gb), 0.0f);
    write_imageh(Y, (int2)(i, y), convert_half4(r));
}

// ---- the Wan 2.2 VAE encoder (model_vaeenc.h) on picture-memory activations ----
// RMS norm over each pixel's channels: x / ||x|| * g (g carries sqrt(C) and the norm's weight), then SiLU if asked
__kernel void rmsni(__read_only image2d_t X, __write_only image2d_t Y, __global const float *g, int Wd, int H, int C4, int silu) {
    const int x = get_global_id(0), y = get_global_id(1);
    if (x >= Wd || y >= H) return;
    float ss = 0.0f;
    for (int c = 0; c < C4; c++) { const float4 v = convert_float4(read_imageh(X, smpi, (int2)(x * C4 + c, y))); ss += dot(v, v); }
    const float r = rsqrt(fmax(ss, 1e-24f));
    for (int c = 0; c < C4; c++) {
        float4 v = convert_float4(read_imageh(X, smpi, (int2)(x * C4 + c, y))) * r * vload4(c, g);
        if (silu) v = v / (1.0f + exp(-v));
        write_imageh(Y, (int2)(x * C4 + c, y), convert_half4(v));
    }
}
// Y = A + B (w texels by h rows)
__kernel void addi(__read_only image2d_t A, __read_only image2d_t B, __write_only image2d_t Y, int w, int h) {
    const int i = get_global_id(0), y = get_global_id(1);
    if (i >= w || y >= h) return;
    write_imageh(Y, (int2)(i, y), convert_half4(convert_float4(read_imageh(A, smpi, (int2)(i, y))) +
                                                convert_float4(read_imageh(B, smpi, (int2)(i, y)))));
}
// one-head attention over all N = Wd x H pixels (q, k, v: C4 texels per pixel each).
// S[i][j] = q_i . k_j * scale
__kernel void attns(__read_only image2d_t Q, __read_only image2d_t K, __global float *S, int Wd, int N, int C4, float scale) {
    const int j = get_global_id(0), i = get_global_id(1);
    if (i >= N || j >= N) return;
    const int xi = i % Wd, yi = i / Wd, xj = j % Wd, yj = j / Wd;
    float d = 0.0f;
    for (int c = 0; c < C4; c++)
        d += dot(convert_float4(read_imageh(Q, smpi, (int2)(xi * C4 + c, yi))),
                 convert_float4(read_imageh(K, smpi, (int2)(xj * C4 + c, yj))));
    S[(size_t)i * N + j] = d * scale;
}
// softmax of each row of S, in place
__kernel void attnsm(__global float *S, int N) {
    const int i = get_global_id(0);
    if (i >= N) return;
    __global float *r = S + (size_t)i * N;
    float m = -1e30f; for (int j = 0; j < N; j++) m = fmax(m, r[j]);
    float t = 0.0f; for (int j = 0; j < N; j++) { const float e = exp(r[j] - m); r[j] = e; t += e; }
    const float inv = 1.0f / t; for (int j = 0; j < N; j++) r[j] *= inv;
}
// O_i = sum_j S[i][j] v_j  (C4 texels per pixel)
__kernel void attno(__read_only image2d_t V, __global const float *S, __write_only image2d_t O, int Wd, int N, int C4) {
    const int c = get_global_id(0), i = get_global_id(1);
    if (c >= C4 || i >= N) return;
    __global const float *r = S + (size_t)i * N;
    float4 a = (float4)(0.0f);
    for (int j = 0; j < N; j++) a += r[j] * convert_float4(read_imageh(V, smpi, (int2)((j % Wd) * C4 + c, j / Wd)));
    write_imageh(O, (int2)((i % Wd) * C4 + c, i / Wd), convert_half4(a));
}
