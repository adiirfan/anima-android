// Anima on the phone's graphics chip. Standard OpenCL 1.2 + cl_khr_fp16, nothing vendor-only.
//
// Activations ("X layout"): image, RGBA32UI, texel (n, k8) = 8 halves = channels 8k8..8k8+7 of token n.
// Packed weights ("W layout"): image, RGBA32UI, texel (og, k2) = outputs 4og..4og+3 at inputs 2k2, 2k2+1.
// The residual stream stays fp32 in a buffer of float4, index og * N + n (channels 4og..4og+3 of token n).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define LD8(img, x, y) as_half8(read_imageui(img, smp, (int2)(x, y)))
#define ST8(img, x, y, v) write_imageui(img, (int2)(x, y), as_uint4(v))

#ifndef TN
#define TN 6
#endif
#ifndef TM
#define TM 2
#endif
#ifndef EPI
#define EPI 0
#endif

// 8 inputs x 4 outputs: x (half8) against w0..w3 (outputs 4og..+3 at inputs 2k, 2k+1). Default: summed in half
// precision, then added to the fp32 total. Old chips (Adreno 630, 2018 driver) flush tiny half products to zero and
// lose precision over 8-product half sums: 5% off after one step. ACC32=4 SCL=64 fixes both (0.1%, ~20% slower);
// SCL 256 already overflows on Anima. ACC32=1: every product in fp32, exact but 8x slower there.
#if ACC32 == 4   // inputs arrive scaled up by SCL (XS, at load: keeps small products out of the range old chips flush
                 // to zero); pairs of products summed in fp16, pairs in fp32; the totals are scaled back once (UNS)
#define XS(v) ((v) * (half)SCL)
#define UNS (1.0f / SCL)
#ifndef HSUM_MASK   // which multiplies take HSUM: bit EPI for gemm/gemml (1 plain, 2 gelu, 4 res+=, 8 res=), 16 the attention blend
#define HSUM_MASK 31
#endif
#ifdef IS_PVSM
#define HSUM_BIT 16
#else
#define HSUM_BIT (1 << EPI)
#endif
#if HSUM == 4 && (HSUM_MASK & HSUM_BIT)   // runs of 4 products in fp16, then fp32: fewer conversions, so faster on Adreno
                // 630. That driver hangs on it unless the kernel is built in a program of its own (kslice), as gemml now is
#define DOT8(x, w0, w1, w2, w3) ( \
    convert_float4(mad((x).s3, (w1).hi, mad((x).s2, (w1).lo, mad((x).s1, (w0).hi, (x).s0 * (w0).lo)))) + \
    convert_float4(mad((x).s7, (w3).hi, mad((x).s6, (w3).lo, mad((x).s5, (w2).hi, (x).s4 * (w2).lo)))))
#else
#define DOT8(x, w0, w1, w2, w3) ( \
    convert_float4(mad((x).s1, (w0).hi, (x).s0 * (w0).lo)) + convert_float4(mad((x).s3, (w1).hi, (x).s2 * (w1).lo)) + \
    convert_float4(mad((x).s5, (w2).hi, (x).s4 * (w2).lo)) + convert_float4(mad((x).s7, (w3).hi, (x).s6 * (w3).lo)))
#endif
#elif ACC32
#define DOT8(x, w0, w1, w2, w3) ( \
    (float)(x).s0 * convert_float4((w0).lo) + (float)(x).s1 * convert_float4((w0).hi) + \
    (float)(x).s2 * convert_float4((w1).lo) + (float)(x).s3 * convert_float4((w1).hi) + \
    (float)(x).s4 * convert_float4((w2).lo) + (float)(x).s5 * convert_float4((w2).hi) + \
    (float)(x).s6 * convert_float4((w3).lo) + (float)(x).s7 * convert_float4((w3).hi))
#else
inline float4 DOT8(half8 x, half8 w0, half8 w1, half8 w2, half8 w3) {
    half4 t = x.s0 * w0.lo;
    t = mad(x.s1, w0.hi, t); t = mad(x.s2, w1.lo, t); t = mad(x.s3, w1.hi, t);
    t = mad(x.s4, w2.lo, t); t = mad(x.s5, w2.hi, t); t = mad(x.s6, w3.lo, t);
    t = mad(x.s7, w3.hi, t);
    return convert_float4(t);
}
#endif
#ifndef XS
#define XS(v) (v)
#define UNS 1.0f
#endif

// Y = X W^T for a batch of matrices (get_global_id(2) = b). Each work item: TN tokens x 4*TM outputs, fp32 sums
// of fp16 products taken 8 inputs at a time.
// EPI 0: Y (X layout) = r          EPI 1: Y = gelu(r)
// EPI 2: res += r * gate           EPI 3: res = r
__kernel void gemm(__read_only image2d_t X, __read_only image2d_t W, int N, int M, int K,
                   int xby,            // X row offset per batch
                   int wbx, int wby,   // W column / row offset per batch
                   int yby,            // Y row offset per batch
                   int xb0, int wb0, int yb0,  // batch index added for X / W / Y (a chunk of heads)
#if EPI < 2
                   __write_only image2d_t Y
#else
                   __global float4 *res, __global const float4 *gate
#endif
                   , int nend   // tokens from nend on are left alone
) {
    const int mb = get_global_id(0), nb = get_global_id(1), b = get_global_id(2);
    const int n0 = nb * TN, og0 = mb * TM;
    if (n0 >= N || og0 * 4 >= M) return;
    const int xr = (b + xb0) * xby, wx = og0 + (b + wb0) * wbx, wr = (b + wb0) * wby;
    float4 r[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; i++)
#pragma unroll
        for (int j = 0; j < TM; j++) r[i][j] = 0;
    for (int k8 = 0; k8 < K / 8; k8++) {
        half8 x[TN];
#pragma unroll
        for (int i = 0; i < TN; i++) x[i] = XS(LD8(X, n0 + i, xr + k8));
#pragma unroll
        for (int j = 0; j < TM; j++) {
            const int wy = wr + k8 * 4;
            const half8 w0 = LD8(W, wx + j, wy), w1 = LD8(W, wx + j, wy + 1), w2 = LD8(W, wx + j, wy + 2),
                        w3 = LD8(W, wx + j, wy + 3);
#pragma unroll
            for (int i = 0; i < TN; i++) {
                r[i][j] += DOT8(x[i], w0, w1, w2, w3);
            }
        }
    }
#if ACC32 == 4
#pragma unroll
    for (int i = 0; i < TN; i++)
#pragma unroll
        for (int j = 0; j < TM; j++) r[i][j] *= UNS;
#endif
#if EPI < 2
    const int yr = (b + yb0) * yby + og0 / 2;
#pragma unroll
    for (int i = 0; i < TN; i++) {
        if (n0 + i >= nend) break;
#pragma unroll
        for (int j = 0; j < TM; j += 2) {
            float8 v = (float8)(r[i][j], r[i][j + 1]);
#if EPI == 1
            v = 0.5f * v * (1.0f + erf(v * 0.70710678f));
#endif
            ST8(Y, n0 + i, yr + j / 2, convert_half8(v));
        }
    }
#else
#pragma unroll
    for (int j = 0; j < TM; j++) {
        const int og = og0 + j;
#if EPI == 2
        const float4 g = gate[og];
#endif
#pragma unroll
        for (int i = 0; i < TN; i++) {
            if (n0 + i >= nend) break;
#if EPI == 2
            res[og * N + n0 + i] += r[i][j] * g;
#else
            res[og * N + n0 + i] = r[i][j];
#endif
        }
    }
#endif
}

// ======== shared-weight multiply (built alone: see kslice in rt.h) ========
// Shared-weight version: a work group of WG items covers WG*TNI tokens x 4*G outputs; the group's weights for KC
// inputs are staged in local memory and read by every item (broadcast). Each item: TNI tokens x 4*G outputs.
// Launch: global (ceil(N/TNI) rounded to WG, M/(4G), batch), local (WG, 1, 1).
#ifndef TNI
#define TNI 2
#endif
#ifndef G
#define G 4
#endif
#ifndef WG
#define WG 128
#endif
#ifndef KC
#define KC 64
#endif
__kernel void gemml(__read_only image2d_t X, __read_only image2d_t W, int N, int M, int K,
                    int xby, int wbx, int wby, int yby, int xb0, int wb0, int yb0,
#if EPI < 2
                    __write_only image2d_t Y
#else
                    __global float4 *res, __global const float4 *gate
#endif
                    , int nend   // tokens from nend on are left alone (one stream of a two-stream model)
) {
    __local half8 wl[KC / 8 * G * 4];
    const int nb = get_global_id(0), mb = get_global_id(1), b = get_global_id(2), lid = get_local_id(0);
    const int n0 = nb * TNI, xr = (b + xb0) * xby, wx = mb * G + (b + wb0) * wbx, wr = (b + wb0) * wby;
    float4 r[TNI][G];
#pragma unroll
    for (int i = 0; i < TNI; i++)
#pragma unroll
        for (int g = 0; g < G; g++) r[i][g] = 0;
    for (int kc = 0; kc < K / 8; kc += KC / 8) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int idx = lid; idx < KC / 8 * G * 4; idx += WG) {
            const int k8l = idx / (G * 4), g = (idx / 4) % G, qq = idx & 3;
            wl[idx] = (kc + k8l < K / 8) ? LD8(W, wx + g, wr + (kc + k8l) * 4 + qq) : (half8)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int kn = min(KC / 8, K / 8 - kc);
        for (int k8l = 0; k8l < kn; k8l++) {
            half8 x[TNI];
#pragma unroll
            for (int i = 0; i < TNI; i++) x[i] = XS(LD8(X, n0 + i, xr + kc + k8l));
#pragma unroll
            for (int g = 0; g < G; g++) {
                const int bi = (k8l * G + g) * 4;
                const half8 w0 = wl[bi], w1 = wl[bi + 1], w2 = wl[bi + 2], w3 = wl[bi + 3];
#pragma unroll
                for (int i = 0; i < TNI; i++) {
                    r[i][g] += DOT8(x[i], w0, w1, w2, w3);
                }
            }
        }
    }
#if ACC32 == 4
#pragma unroll
    for (int i = 0; i < TNI; i++)
#pragma unroll
        for (int g = 0; g < G; g++) r[i][g] *= UNS;
#endif
#pragma unroll
    for (int i = 0; i < TNI; i++) {
        const int n = n0 + i;
        if (n >= nend) break;
#if EPI < 2
#pragma unroll
        for (int g = 0; g < G; g += 2) {
            float8 v = (float8)(r[i][g], r[i][g + 1]);
#if EPI == 1 && !GELU_PASS   // GELU_PASS: gelu8 applies it afterwards (the erf here lowers how many threads Adreno
            v = 0.5f * v * (1.0f + erf(v * 0.70710678f));   // 630 runs at once, which undoes the HSUM speed-up)
#endif
            ST8(Y, n, (b + yb0) * yby + (mb * G + g) / 2, convert_half8(v));
        }
#else
#pragma unroll
        for (int g = 0; g < G; g++) {
            const int og = mb * G + g;
#if EPI == 2
            res[og * N + n] += r[i][g] * gate[og];
#else
            res[og * N + n] = r[i][g];
#endif
        }
#endif
    }
}

// ======== end of shared-weight multiply ========

// out = layernorm(res) * (1 + sc) + sh, one work item per token, written in X layout
__kernel void lnmod(__global const float4 *res, __global const float4 *sh, __global const float4 *sc,
                    __write_only image2d_t Y, int N, int D, float eps) {
    const int n = get_global_id(0);
    if (n >= N) return;
    const int D4 = D / 4;
    float4 s = 0;
    for (int c = 0; c < D4; c++) s += res[c * N + n];
    const float mean = (s.x + s.y + s.z + s.w) / D;
    float4 v = 0;
    for (int c = 0; c < D4; c++) { float4 d = res[c * N + n] - mean; v = mad(d, d, v); }
    const float rstd = rsqrt((v.x + v.y + v.z + v.w) / D + eps);
    for (int c = 0; c < D4; c += 2) {
        float4 a = (res[c * N + n] - mean) * rstd * (1.0f + sc[c]) + sh[c];
        float4 b = (res[(c + 1) * N + n] - mean) * rstd * (1.0f + sc[c + 1]) + sh[c + 1];
        ST8(Y, n, c / 2, convert_half8((float8)(a, b)));
    }
}

// same as lnmod, but a work group of 16 tokens x LN lanes shares each token's sums through local memory.
// Launch: global (N rounded to 16, LN), local (16, LN).
#ifndef LN
#define LN 16
#endif
__kernel void lnmod2(__global const float4 *res, __global const float4 *sh, __global const float4 *sc,
                     __write_only image2d_t Y, int N, int D, float eps) {
    __local float red[LN][16];
    const int t = get_local_id(0), l = get_local_id(1), n = get_global_id(0);
    const int D4 = D / 4, nn = min(n, N - 1);
    float4 s = 0;
    for (int c = l; c < D4; c += LN) s += res[c * N + nn];
    red[l][t] = s.x + s.y + s.z + s.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    float tot = 0;
    for (int i = 0; i < LN; i++) tot += red[i][t];
    const float mean = tot / D;
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 v = 0;
    for (int c = l; c < D4; c += LN) { float4 d = res[c * N + nn] - mean; v = mad(d, d, v); }
    red[l][t] = v.x + v.y + v.z + v.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    tot = 0;
    for (int i = 0; i < LN; i++) tot += red[i][t];
    const float rstd = rsqrt(tot / D + eps);
    if (n >= N) return;
    for (int k8 = l; k8 < D / 8; k8 += LN) {
        const int c = 2 * k8;
        float4 a = (res[c * N + n] - mean) * rstd * (1.0f + sc[c]) + sh[c];
        float4 b = (res[(c + 1) * N + n] - mean) * rstd * (1.0f + sc[c + 1]) + sh[c + 1];
        ST8(Y, n, k8, convert_half8((float8)(a, b)));
    }
}

// Attention inputs from a projection output in X layout. One work item = 4 tokens x 1 head (128 channels).
//   q: rms * qw * qscale, rope (if ROPE), written in X layout to Qo (n, h*16 + d8)          (when DO_Q)
//   k: rms * kw, rope, written in W layout to Ko: texel (n/4, h*64 + d/2)                    (when DO_K)
//   v: written in W layout to Vo: texel (h*32 + d/4, n/2)                                    (when DO_V)
// qr / kr / vr: X-layout row where q / k / v channel 0 sits in the input
#ifndef QKFULL
#define QKFULL 0
#endif
#ifndef KVR
#define KVR 1   // query heads per key / value head (Krea: 48 queries share 12 key / value heads)
#endif
__kernel void attnprep(__read_only image2d_t In, int N, int qr, int kr, int vr,
                       __global const float *qw, __global const float *kw, float qscale,
                       __global const float *P, __global const float *Q,
                       __write_only image2d_t Qo, __write_only image2d_t Ko, __write_only image2d_t Vo
#if QKFULL
                       , __global const float *rq, __global const float *rkf   // whole-width 1/rms per token
#endif
                       ) {
    const int n0 = get_global_id(0) * 4, h = get_global_id(1), hk = h / KVR;
    if (n0 >= N) return;
#if QKFULL
    qw += h * 128; kw += h * 128;
#endif
#if DO_Q
    for (int t = 0; t < 4 && n0 + t < N; t++) {
        const int n = n0 + t;
        float ss = 0;
        for (int c = 0; c < 16; c++) { float8 a = convert_float8(LD8(In, n, qr + h * 16 + c)); ss += dot(a.lo, a.lo) + dot(a.hi, a.hi); }
#if QKFULL
        const float rs = rq[n] * qscale;
#else
        const float rs = rsqrt(ss / 128 + 1e-6f) * qscale;
#endif
        for (int c = 0; c < 16; c++) {
            float8 a = convert_float8(LD8(In, n, qr + h * 16 + c)) * vload8(c, qw) * rs;
#if ROPE
            const int cs = c ^ 8;
            float8 p = convert_float8(LD8(In, n, qr + h * 16 + cs)) * vload8(cs, qw) * rs;
            a = a * vload8(c, P + n * 128) + p * vload8(c, Q + n * 128);
#endif
            ST8(Qo, n, h * 16 + c, convert_half8(a));
        }
    }
#endif
#if DO_K
    float rk[4];
    for (int t = 0; t < 4; t++) {
        float ss = 0;
        for (int c = 0; c < 16; c++) { float8 a = convert_float8(LD8(In, n0 + t, kr + hk * 16 + c)); ss += dot(a.lo, a.lo) + dot(a.hi, a.hi); }
#if QKFULL
        rk[t] = rkf[min(n0 + t, N - 1)];
#else
        rk[t] = rsqrt(ss / 128 + 1e-6f);
#endif
    }
    for (int c = 0; c < 16; c++) {
        float8 a[4];
        for (int t = 0; t < 4; t++) {
            const int n = min(n0 + t, N - 1);
            a[t] = convert_float8(LD8(In, n0 + t, kr + hk * 16 + c)) * vload8(c, kw) * rk[t];
#if ROPE
            const int cs = c ^ 8;
            float8 p = convert_float8(LD8(In, n0 + t, kr + hk * 16 + cs)) * vload8(cs, kw) * rk[t];
            a[t] = a[t] * vload8(c, P + n * 128) + p * vload8(c, Q + n * 128);
#endif
        }
        // 8 channels of 4 tokens -> 4 texels, each 4 tokens x 2 channels
        ST8(Ko, n0 / 4, h * 64 + c * 4 + 0, convert_half8((float8)(a[0].s0, a[1].s0, a[2].s0, a[3].s0, a[0].s1, a[1].s1, a[2].s1, a[3].s1)));
        ST8(Ko, n0 / 4, h * 64 + c * 4 + 1, convert_half8((float8)(a[0].s2, a[1].s2, a[2].s2, a[3].s2, a[0].s3, a[1].s3, a[2].s3, a[3].s3)));
        ST8(Ko, n0 / 4, h * 64 + c * 4 + 2, convert_half8((float8)(a[0].s4, a[1].s4, a[2].s4, a[3].s4, a[0].s5, a[1].s5, a[2].s5, a[3].s5)));
        ST8(Ko, n0 / 4, h * 64 + c * 4 + 3, convert_half8((float8)(a[0].s6, a[1].s6, a[2].s6, a[3].s6, a[0].s7, a[1].s7, a[2].s7, a[3].s7)));
    }
#endif
#if DO_V
    for (int c = 0; c < 16; c++) {
        half8 a0 = LD8(In, n0, vr + hk * 16 + c), a1 = LD8(In, n0 + 1, vr + hk * 16 + c);
        half8 a2 = LD8(In, n0 + 2, vr + hk * 16 + c), a3 = LD8(In, n0 + 3, vr + hk * 16 + c);
        ST8(Vo, h * 32 + c * 2, n0 / 2, (half8)(a0.lo, a1.lo));
        ST8(Vo, h * 32 + c * 2 + 1, n0 / 2, (half8)(a0.hi, a1.hi));
        ST8(Vo, h * 32 + c * 2, n0 / 2 + 1, (half8)(a2.lo, a3.lo));
        ST8(Vo, h * 32 + c * 2 + 1, n0 / 2 + 1, (half8)(a2.hi, a3.hi));
    }
#endif
}

// Keys and values on a half-resolution grid (token downsampling, "ToDo", Smith et al. 2024): every 2x2 group of
// neighbouring tokens becomes one key (the top-left token's, rope exact) and one value (the average of the four); queries stay
// full. Self-attention then compares each token with N/4 keys. Same layouts as attnprep (Ko, Vo) with N2 = N/4 keys.
// Launch (N2/4, heads). GW: tokens per row of the full grid.
__kernel void kvdown(__read_only image2d_t In, int N2, int GW, int kr, int vr, __global const float *kw,
                     __global const float *P, __global const float *Q,
                     __write_only image2d_t Ko, __write_only image2d_t Vo) {
    const int m0 = get_global_id(0) * 4, h = get_global_id(1);
    if (m0 >= N2) return;
    const int GW2 = GW / 2;
    int src[4][4];
    float rk[4][4];
    for (int t = 0; t < 4; t++) {
        const int m = min(m0 + t, N2 - 1), y = m / GW2, x = m % GW2;
        src[t][0] = 2 * y * GW + 2 * x; src[t][1] = src[t][0] + 1; src[t][2] = src[t][0] + GW; src[t][3] = src[t][2] + 1;
        for (int u = 0; u < 4; u++) {
            float ss = 0;
            for (int c = 0; c < 16; c++) { float8 a = convert_float8(LD8(In, src[t][u], kr + h * 16 + c)); ss += dot(a.lo, a.lo) + dot(a.hi, a.hi); }
            rk[t][u] = rsqrt(ss / 128 + 1e-6f);
        }
    }
    for (int c = 0; c < 16; c++) {
        const int cs = c ^ 8;
        float8 a[4];
        for (int t = 0; t < 4; t++) {
            a[t] = 0;
#if KAVG
            // keys averaged after rope: smears the position stamp (tested 2026-11-18: a featureless blur)
            for (int u = 0; u < 4; u++) {
                const int n = src[t][u];
                const float8 v = convert_float8(LD8(In, n, kr + h * 16 + c)) * vload8(c, kw) * rk[t][u];
                const float8 p = convert_float8(LD8(In, n, kr + h * 16 + cs)) * vload8(cs, kw) * rk[t][u];
                a[t] += v * vload8(c, P + n * 128) + p * vload8(c, Q + n * 128);
            }
            a[t] *= 0.25f;
#else
            // the key of the group's top-left token, so its rope position stays exact; values are still averaged
            {
                const int n = src[t][0];
                const float8 v = convert_float8(LD8(In, n, kr + h * 16 + c)) * vload8(c, kw) * rk[t][0];
                const float8 p = convert_float8(LD8(In, n, kr + h * 16 + cs)) * vload8(cs, kw) * rk[t][0];
                a[t] = v * vload8(c, P + n * 128) + p * vload8(c, Q + n * 128);
            }
#endif
        }
        ST8(Ko, m0 / 4, h * 64 + c * 4 + 0, convert_half8((float8)(a[0].s0, a[1].s0, a[2].s0, a[3].s0, a[0].s1, a[1].s1, a[2].s1, a[3].s1)));
        ST8(Ko, m0 / 4, h * 64 + c * 4 + 1, convert_half8((float8)(a[0].s2, a[1].s2, a[2].s2, a[3].s2, a[0].s3, a[1].s3, a[2].s3, a[3].s3)));
        ST8(Ko, m0 / 4, h * 64 + c * 4 + 2, convert_half8((float8)(a[0].s4, a[1].s4, a[2].s4, a[3].s4, a[0].s5, a[1].s5, a[2].s5, a[3].s5)));
        ST8(Ko, m0 / 4, h * 64 + c * 4 + 3, convert_half8((float8)(a[0].s6, a[1].s6, a[2].s6, a[3].s6, a[0].s7, a[1].s7, a[2].s7, a[3].s7)));
    }
    for (int c = 0; c < 16; c++) {
        float8 v[4];
        for (int t = 0; t < 4; t++) {
            v[t] = 0;
            for (int u = 0; u < 4; u++) v[t] += convert_float8(LD8(In, src[t][u], vr + h * 16 + c));
            v[t] *= 0.25f;
        }
        const half8 a0 = convert_half8(v[0]), a1 = convert_half8(v[1]), a2 = convert_half8(v[2]), a3 = convert_half8(v[3]);
        ST8(Vo, h * 32 + c * 2, m0 / 2, (half8)(a0.lo, a1.lo));
        ST8(Vo, h * 32 + c * 2 + 1, m0 / 2, (half8)(a0.hi, a1.hi));
        ST8(Vo, h * 32 + c * 2, m0 / 2 + 1, (half8)(a2.lo, a3.lo));
        ST8(Vo, h * 32 + c * 2 + 1, m0 / 2 + 1, (half8)(a2.hi, a3.hi));
    }
}

// softmax over keys for one (query, head): scores S (X layout, row h*Lk/8 + j) -> probabilities Pr (same layout).
// bias is added to the last key (it stands in for the blank prompt rows that were not computed).
__kernel void softmax(__read_only image2d_t S, __write_only image2d_t Pr, int N, int Lk, float bias) {
    const int n = get_global_id(0), h = get_global_id(1);
    if (n >= N) return;
    const int r0 = h * (Lk / 8), J = Lk / 8;
    float m = -INFINITY;
    for (int j = 0; j < J; j++) {
        float8 s = convert_float8(LD8(S, n, r0 + j));
        if (j == J - 1) s.s7 += bias;
        m = fmax(m, fmax(fmax(fmax(s.s0, s.s1), fmax(s.s2, s.s3)), fmax(fmax(s.s4, s.s5), fmax(s.s6, s.s7))));
    }
    float sum = 0;
    for (int j = 0; j < J; j++) {
        float8 s = convert_float8(LD8(S, n, r0 + j));
        if (j == J - 1) s.s7 += bias;
        float8 e = exp(s - m);
        sum += dot(e.lo, (float4)1) + dot(e.hi, (float4)1);
    }
    const float inv = 1.0f / sum;
    for (int j = 0; j < J; j++) {
        float8 s = convert_float8(LD8(S, n, r0 + j));
        if (j == J - 1) s.s7 += bias;
        ST8(Pr, n, r0 + j, convert_half8(exp(s - m) * inv));
    }
}

// softmax with SL lanes per (query, head) row: each lane keeps its share of the row in registers, so the scores are
// read once. Launch: global (SL, N, heads), local (SL, SR, 1).  Needs Lk / 8 <= SL * MAXT.
#ifndef SL
#define SL 16
#endif
#ifndef SR
#define SR 8
#endif
#ifndef MAXT
#define MAXT 8
#endif
__kernel void softmax2(__read_only image2d_t S, __write_only image2d_t Pr, int N, int Lk, float bias, int kvalid) {
    __local float red[SR][SL];
    const int l = get_local_id(0), rr = get_local_id(1), n = get_global_id(1), h = get_global_id(2);
    const int r0 = h * (Lk / 8), J = Lk / 8;
    float8 v[MAXT];
    float m = -INFINITY;
#pragma unroll
    for (int t = 0; t < MAXT; t++) {
        const int j = l + t * SL;
        v[t] = (j < J) ? convert_float8(LD8(S, n, r0 + j)) : (float8)(-INFINITY);
        if (j == J - 1) v[t].s7 += bias;
        if ((j + 1) * 8 > kvalid) {   // keys from kvalid on are padding
            const int8 e = (int8)(0, 1, 2, 3, 4, 5, 6, 7) + j * 8;
            v[t] = select(v[t], (float8)(-INFINITY), e >= (int8)(kvalid));
        }
        m = fmax(m, fmax(fmax(fmax(v[t].s0, v[t].s1), fmax(v[t].s2, v[t].s3)), fmax(fmax(v[t].s4, v[t].s5), fmax(v[t].s6, v[t].s7))));
    }
    red[rr][l] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = red[rr][0];
    for (int i = 1; i < SL; i++) m = fmax(m, red[rr][i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0;
#pragma unroll
    for (int t = 0; t < MAXT; t++) {
        v[t] = exp(v[t] - m);
        sum += dot(v[t].lo, (float4)1) + dot(v[t].hi, (float4)1);
    }
    red[rr][l] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    sum = 0;
    for (int i = 0; i < SL; i++) sum += red[rr][i];
    const float inv = 1.0f / sum;
    if (n >= N) return;
#pragma unroll
    for (int t = 0; t < MAXT; t++) {
        const int j = l + t * SL;
        if (j < J) ST8(Pr, n, r0 + j, convert_half8(v[t] * inv));
    }
}

// q for attention: one work item per (token, head): rms * qw * qscale, rope when ROPE. Launch (N, heads).
__kernel void prepq(__read_only image2d_t In, int N, int qr, __global const float *qw, float qscale,
                    __global const float *P, __global const float *Q, __write_only image2d_t Qo
#if QKFULL
                    , __global const float *rq
#endif
                    ) {
    const int n = get_global_id(0), h = get_global_id(1);
    if (n >= N) return;
#if QKFULL
    qw += h * 128;
#endif
    float8 a[16];
    float ss = 0;
#pragma unroll
    for (int c = 0; c < 16; c++) { a[c] = convert_float8(LD8(In, n, qr + h * 16 + c)); ss += dot(a[c].lo, a[c].lo) + dot(a[c].hi, a[c].hi); }
#if QKFULL
    const float rs = rq[n] * qscale;
#else
    const float rs = rsqrt(ss / 128 + 1e-6f) * qscale;
#endif
#pragma unroll
    for (int c = 0; c < 16; c++) a[c] *= vload8(c, qw) * rs;
#pragma unroll
    for (int c = 0; c < 16; c++) {
        float8 o = a[c];
#if ROPE
        o = o * vload8(c, P + n * 128) + a[c ^ 8] * vload8(c, Q + n * 128);
#endif
        ST8(Qo, n, h * 16 + c, convert_half8(o));
    }
}

// out[n] = 1 / rms of token n's channels in texel columns [c0, c0 + C8) (whole-width q / k norms). Launch (64, N), local (64, 1).
__kernel void rowrms(__read_only image2d_t In, int N, int c0, int C8, __global float *out) {
    __local float red[64];
    const int l = get_local_id(0), n = get_global_id(1);
    float ss = 0;
    if (n < N) for (int c = l; c < C8; c += 64) { float8 a = convert_float8(LD8(In, n, c0 + c)); ss += dot(a.lo, a.lo) + dot(a.hi, a.hi); }
    red[l] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = 32; w > 0; w >>= 1) { if (l < w) red[l] += red[l + w]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0 && n < N) out[n] = rsqrt(red[0] / (C8 * 8) + 1e-6f);
}

// y[m] = W[m] . act(x) + add[m]   (W fp16 row-major [M][K]); act: silu = 1 SiLU, 2 GELU (tanh), 0 none
__kernel void gemv(__global const half *W, __global const float *x, __global const float *add,
                   __global float *y, int M, int K, int silu) {
    const int m = get_global_id(0);
    if (m >= M) return;
    float s = 0;
    const __global half *w = W + (size_t)m * K;
    for (int k = 0; k < K; k += 8) {
        float8 a = vload8(0, x + k);
        if (silu == 1) a = a / (1.0f + exp(-a));
        else if (silu == 2) a = 0.5f * a * (1.0f + tanh(0.7978845608f * (a + 0.044715f * a * a * a)));   // GELU (tanh)
        s += dot(convert_float8(vload8(0, w + k)).lo, a.lo) + dot(convert_float8(vload8(0, w + k)).hi, a.hi);
    }
    y[m] = s + (add ? add[m] : 0.0f);
}

// Krea's text fusion: attention inside groups of G consecutive rows (the 12 reader layers of one prompt token, or all
// prompt rows), q / k rms-normed with qn / kn (eps 1e-5), no rope. In: X layout, q / k / v at texel columns qr / kr /
// vr, 16 texels (128 channels) per head. One work item per (row, head): launch (n1, heads).
__kernel void grpattn(__read_only image2d_t In, int n1, int grp, int qr, int kr, int vr,
                      __global const float *qw, __global const float *kw, __write_only image2d_t O) {
    const int n = get_global_id(0), h = get_global_id(1);
    if (n >= n1) return;
    const int g0 = n / grp * grp, g1 = min(g0 + grp, n1);
    float8 qv[16], acc[16];
    float ss = 0;
    for (int c = 0; c < 16; c++) { qv[c] = convert_float8(LD8(In, n, qr + h * 16 + c)); ss += dot(qv[c].lo, qv[c].lo) + dot(qv[c].hi, qv[c].hi); }
    const float rq = rsqrt(ss / 128 + 1e-5f) * 0.08838834764f;   // and 1 / sqrt(128)
    for (int c = 0; c < 16; c++) { qv[c] *= vload8(c, qw) * rq; acc[c] = (float8)0.0f; }
    float m = -INFINITY, l = 0;
    for (int j = g0; j < g1; j++) {
        float kk = 0, s = 0;
        for (int c = 0; c < 16; c++) {
            const float8 a = convert_float8(LD8(In, j, kr + h * 16 + c));
            kk += dot(a.lo, a.lo) + dot(a.hi, a.hi);
            const float8 b = a * vload8(c, kw) * qv[c];
            s += b.s0 + b.s1 + b.s2 + b.s3 + b.s4 + b.s5 + b.s6 + b.s7;
        }
        s *= rsqrt(kk / 128 + 1e-5f);
        const float mn = fmax(m, s), corr = exp(m - mn), p = exp(s - mn);
        l = l * corr + p;
        for (int c = 0; c < 16; c++) acc[c] = acc[c] * corr + p * convert_float8(LD8(In, j, vr + h * 16 + c));
        m = mn;
    }
    for (int c = 0; c < 16; c++) ST8(O, n, h * 16 + c, convert_half8(acc[c] / l));
}
// exact GELU of an X-layout image, tokens [n0, n): the second half of a split GELU multiply (ops.h GELUP).
// launch (n - n0, C / 8)
__kernel void gelu8(__read_only image2d_t Y, int n0, int n, __write_only image2d_t Z) {
    const int t = n0 + get_global_id(0), c = get_global_id(1);
    if (t >= n) return;
    const float8 a = convert_float8(LD8(Y, t, c));
    ST8(Z, t, c, convert_half8(0.5f * a * (1.0f + erf(a * 0.70710678f))));
}
// Krea's text MLP: Z = gelu_tanh(Y + b), X layout. launch (N, C / 8)
__kernel void biasgelu(__read_only image2d_t Y, __global const float *b, int N, __write_only image2d_t Z) {
    const int n = get_global_id(0), c = get_global_id(1);
    if (n >= N) return;
    const float8 a = convert_float8(LD8(Y, n, c)) + vload8(c, b);
    ST8(Z, n, c, convert_half8(0.5f * a * (1.0f + tanh(0.7978845608f * (a + 0.044715f * a * a * a)))));
}
// Krea: the attention output times sigmoid(gate), the gate at texel column gc of G (both X layout, texel (n, c8)).
// launch (N, Dm / 8)
__kernel void sgate(__read_only image2d_t O, __read_only image2d_t Gt, int gc, int N, __write_only image2d_t Y) {
    const int n = get_global_id(0), c = get_global_id(1);
    if (n >= N) return;
    const float8 o = convert_float8(LD8(O, n, c)), g = convert_float8(LD8(Gt, n, gc + c));
    ST8(Y, n, c, convert_half8(o / (1.0f + exp(-g))));
}

// SwiGLU after a 16-bit multiply (chips without the byte dot product): Y holds the stored rows [gate 8][up 8] per 16
// outputs (X layout, texel (n, c8)), Z gets silu(gate) * up. Launch (N, M / 16) for M stored rows.
__kernel void swiglu16(__read_only image2d_t Y, int N, __write_only image2d_t Z) {
    const int n = get_global_id(0), c = get_global_id(1);
    if (n >= N) return;
    const float8 g = convert_float8(LD8(Y, n, 2 * c)), u = convert_float8(LD8(Y, n, 2 * c + 1));
    ST8(Z, n, c, convert_half8(g / (1.0f + exp(-g)) * u));
}

// res (channel groups first: res[c * N + n]) += rms_norm(A[n]) * w, A an X-layout image (n, c8) of Dm channels.
// Fibo's norms after a layer's output ("sandwich" norms) with the gate folded into w. One work group of 64 per token:
// launch (64, rows), local (64, 1), offset the first row on dimension 1.
__kernel void rmsadd(__read_only image2d_t A, __global float4 *res, __global const float4 *w, int N, int Dm) {
    const int l = get_local_id(0), n = get_global_id(1);
    __local float red[64];
    float s = 0.0f;
    for (int c8 = l; c8 < Dm / 8; c8 += 64) { const float8 v = convert_float8(LD8(A, n, c8)); s += dot(v.lo, v.lo) + dot(v.hi, v.hi); }
    red[l] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 32; k > 0; k >>= 1) { if (l < k) red[l] += red[l + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float r = rsqrt(red[0] / Dm + 1e-6f);
    for (int c8 = l; c8 < Dm / 8; c8 += 64) {
        const float8 v = convert_float8(LD8(A, n, c8)) * r;
        res[(size_t)(2 * c8) * N + n] += v.lo * w[2 * c8];
        res[(size_t)(2 * c8 + 1) * N + n] += v.hi * w[2 * c8 + 1];
    }
}

// ======== 8-bit path ========
// Activations for the integer multiply: image RGBA32UI, texel (n, k16) = 16 x int8 of token n, with one fp32 scale
// per token and block of GS inputs (sx[n * K/GS + b]). Before rounding, every 256 inputs are rotated by the regular
// Hadamard (ConvRot) that the packed weights already carry, which spreads extreme channels across the block.
#ifndef GS
#define GS 64
#endif
#ifndef CEN
#define CEN 0
#endif

// 64 lanes per token, QT tokens per work group (local (64, QT)). SRC 0: input is an fp16 X-layout image (attention
// output, MLP hidden). SRC 1: input is layernorm(res) * (1 + sc) + sh (the residual stream). Launch (64, N).
#ifndef QT
#define QT 4
#endif
#ifndef HOUT
#define HOUT 0
#endif
__kernel void rotq(
#if SRC == 1
    __global const float4 *res, __global const float4 *sh, __global const float4 *sc, float eps,
#else
    __read_only image2d_t In,
#endif
    int N, int K, __write_only image2d_t Q, __global float *sx
#if CEN
    , __global const float4 *mu   // CEN: per-channel input average, taken off before rounding (added back as W.mu)
#endif
    ) {
    __local float bufa[QT][256], reda[QT][64];
    __local uint ua[QT][64];
    const int l = get_local_id(0), t = get_local_id(1), n0 = get_global_id(1), n = min(n0, N - 1);
    __local float *buf = bufa[t], *red = reda[t];
    __local uint *u = ua[t];
#if SRC == 1
#if RMSN   // RMS norm (MOSS): no mean taken out
    const float mean = 0.0f;
#else
    float4 s4 = 0;
    for (int c = l; c < K / 4; c += 64) s4 += res[c * N + n];
    red[l] = s4.x + s4.y + s4.z + s4.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = 32; w > 0; w >>= 1) { if (l < w) red[l] += red[l + w]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float mean = red[0] / K;
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    float4 v4 = 0;
    for (int c = l; c < K / 4; c += 64) { float4 d = res[c * N + n] - mean; v4 = mad(d, d, v4); }
    red[l] = v4.x + v4.y + v4.z + v4.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = 32; w > 0; w >>= 1) { if (l < w) red[l] += red[l + w]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float rstd = rsqrt(red[0] / K + eps);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    const int NB = K / GS, LB = GS / 4;
    for (int g = 0; g < K / 256; g++) {
#if SRC == 1
        const int c4 = g * 64 + l;
        float4 v = (res[c4 * N + n] - mean) * rstd * (1.0f + sc[c4]) + sh[c4];
#else
        const half8 t8 = LD8(In, n, g * 32 + l / 2);
        float4 v = convert_float4((l & 1) ? t8.hi : t8.lo);
#endif
#if CEN
        v -= mu[g * 64 + l];
#endif
#if ROT
        vstore4(v, l, buf);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 1; s < 256; s *= 4) {
            const int i0 = (l / s) * s * 4 + (l % s);
            const float a0 = buf[i0], a1 = buf[i0 + s], a2 = buf[i0 + 2 * s], a3 = buf[i0 + 3 * s];
            buf[i0] = a0 + a1 + a2 - a3;
            buf[i0 + s] = a0 + a1 - a2 + a3;
            buf[i0 + 2 * s] = a0 - a1 + a2 + a3;
            buf[i0 + 3 * s] = -a0 + a1 + a2 + a3;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        v = vload4(l, buf) * (1.0f / 16.0f);
#endif
#if HOUT   // A16 (see deq8): the rotated inputs stay fp16, texel (n, k8)
        barrier(CLK_LOCAL_MEM_FENCE);
        vstore4(v, l, buf);
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((l & 1) == 0 && n0 < N) ST8(Q, n, g * 32 + l / 2, convert_half8(vload8(l / 2, buf)));
        barrier(CLK_LOCAL_MEM_FENCE);
#else
        red[l] = fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w)));
        barrier(CLK_LOCAL_MEM_FENCE);
        const int b0 = (l / LB) * LB;
        float m = 1e-12f;
        for (int i = 0; i < LB; i++) m = fmax(m, red[b0 + i]);
        u[l] = as_uint(convert_char4_sat(convert_int4_rte(v * (127.0f / m))));
        if (l == b0 && n0 < N) sx[n * NB + (g * 256 + 4 * l) / GS] = m / 127.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((l & 3) == 0 && n0 < N) write_imageui(Q, (int2)(n, g * 16 + l / 4), (uint4)(u[l], u[l + 1], u[l + 2], u[l + 3]));
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }
}

// A16 (chips without the byte dot product): a layer's 8-bit weights turned into the fp16 W layout just before it runs,
// so the tuned fp16 multiply does the work (rotq HOUT gives it fp16 inputs). One item per 4 outputs x 16 inputs.
// W8 texel (o - part*MW, part*(K/16) + k16) = 16 x int8 inputs of output o, scale sw[o * K/GS + k/GS];
// Wp texel (og, k2) = outputs 4og..+3 at input 2k2, then at 2k2 + 1. Launch (M/4, K/16).
inline uint4 dq4(uint4 u, int k16) { return (k16 & 1) ? (u & (uint4)0xF0F0F0F0u) : ((u << 4) & (uint4)0xF0F0F0F0u); }
__kernel void deq8(__read_only image2d_t W8, __global const float *sw, int M, int K, __write_only image2d_t Wp) {
    const int og = get_global_id(0), k16 = get_global_id(1);
    if (og * 4 >= M || k16 * 16 >= K) return;
    const int MW = get_image_width(W8), NB = K / GS, o = og * 4, sb = k16 * 16 / GS;
#if defined(W4) && W4   // 4-bit table (texel (o, k32): inputs 32b + j low half, 32b + 16 + j high half), read as gemmq does
#define DQR(oj) dq4(read_imageui(W8, smp, (int2)((oj) - ((oj) / MW) * MW, ((oj) / MW) * (K / 32) + k16 / 2)), k16)
#else
#define DQR(oj) read_imageui(W8, smp, (int2)((oj) - ((oj) / MW) * MW, ((oj) / MW) * (K / 16) + k16))
#endif
#define DQL(j) convert_half16(convert_float16(as_char16(DQR(o + j))) * sw[(o + j) * NB + sb])
    const half16 h0 = DQL(0), h1 = DQL(1), h2 = DQL(2), h3 = DQL(3);
#define DQP(p, a, b) ST8(Wp, og, k16 * 8 + p, (half8)(h0.a, h1.a, h2.a, h3.a, h0.b, h1.b, h2.b, h3.b))
    DQP(0, s0, s1); DQP(1, s2, s3); DQP(2, s4, s5); DQP(3, s6, s7);
    DQP(4, s8, s9); DQP(5, sa, sb); DQP(6, sc, sd); DQP(7, se, sf);
#undef DQL
#undef DQR
#undef DQP
}

// Integer multiply: X (n, k16) int8 x W (o, k16) int8 -> int32 per block of GS inputs, scaled into fp32 totals.
// A work group of QWG items shares QOB outputs' weights through local memory; each item QTN tokens x QOB outputs.
// Launch: global (ceil(N/QTN) rounded to QWG, M/QOB), local (QWG, 1). Epilogues as gemm (EPI 0..3).
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#elif !defined(NO_IDP)
#define NO_IDP 1   // older drivers (e.g. Adreno 630, 2018) do not have it at all: the pragma alone fails the build
#endif
#if NO_IDP   // chips without the 4 x int8 dot product (the computer's Radeon, for testing): plain arithmetic
inline int dot_acc_sat_4x8packed_ss_int(uint a, uint b, int acc) {
    const char4 x = as_char4(a), y = as_char4(b);
    return acc + x.s0 * y.s0 + x.s1 * y.s1 + x.s2 * y.s2 + x.s3 * y.s3;
}
#endif
#ifndef QTN
#define QTN 2
#endif
#ifndef QOB
#define QOB 16
#endif
#ifndef QWG
#define QWG 128
#endif
#ifndef W4
#define W4 0
#endif
#ifndef QKC
#define QKC (GS > 128 ? GS : 128)   // inputs staged per pass: whole scale blocks
#endif
// EPI 4: SwiGLU. The weight rows are stored [gate 8][up 8] per 16 outputs, so each item can finish
// silu(gate) * up itself and write M/2 outputs.
__kernel void gemmq(__read_only image2d_t X, __read_only image2d_t W, int N, int M, int K,
                    __global const float *sx, __global const float *sw,
#if EPI < 2 || EPI == 4
                    __write_only image2d_t Y
#else
                    __global float4 *res, __global const float4 *gate
#endif
#if CEN
                    , __global const float *bias   // CEN: W.mu of the centred input, per output
#endif
                    , int nend   // tokens from nend on are left alone (one stream of a two-stream model)
) {
    __local uint4 wl[QKC / 16 * QOB];
    __local float swl_all[QKC / GS * QOB];
    const int nb = get_global_id(0), mb = get_global_id(1), lid = get_local_id(0);
    const int n0 = nb * QTN, o0 = mb * QOB, NB = K / GS, MW = get_image_width(W);
    int acc[QTN][QOB];
    float f[QTN][QOB];
#pragma unroll
    for (int i = 0; i < QTN; i++)
#pragma unroll
        for (int o = 0; o < QOB; o++) { acc[i][o] = 0; f[i][o] = 0; }
    for (int kc = 0; kc < K / 16; kc += QKC / 16) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int idx = lid; idx < QKC / 16 * QOB; idx += QWG) {
            // outputs past the image width live in the next part, stacked below (see q8img on the host)
            const int o = o0 + idx % QOB, part = o / MW;
#if W4
            // 4-bit: texel (o, k32) holds inputs 32b + j in byte j's low half and 32b + 16 + j in its high half; kept
            // as nibble * 16 (an int8 the dot product takes as is; the scales carry the 1/16)
            const int k16 = kc + idx / QOB;
            const uint4 u = read_imageui(W, smp, (int2)(o - part * MW, part * (K / 32) + k16 / 2));
            wl[idx] = (k16 & 1) ? (u & (uint4)0xF0F0F0F0u) : ((u << 4) & (uint4)0xF0F0F0F0u);
#else
            wl[idx] = read_imageui(W, smp, (int2)(o - part * MW, part * (K / 16) + kc + idx / QOB));
#endif
        }
        for (int idx = lid; idx < QKC / GS * QOB; idx += QWG) swl_all[idx] = sw[(o0 + idx % QOB) * NB + kc / (GS / 16) + idx / QOB];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kl = 0; kl < QKC / 16; kl++) {
            uint4 x[QTN];
#pragma unroll
            for (int i = 0; i < QTN; i++) x[i] = read_imageui(X, smp, (int2)(n0 + i, kc + kl));
#pragma unroll
            for (int o = 0; o < QOB; o++) {
                const uint4 w = wl[kl * QOB + o];
#pragma unroll
                for (int i = 0; i < QTN; i++) {
                    int a = acc[i][o];
                    a = dot_acc_sat_4x8packed_ss_int(x[i].x, w.x, a); a = dot_acc_sat_4x8packed_ss_int(x[i].y, w.y, a);
                    a = dot_acc_sat_4x8packed_ss_int(x[i].z, w.z, a); a = dot_acc_sat_4x8packed_ss_int(x[i].w, w.w, a);
                    acc[i][o] = a;
                }
            }
            if (((kc + kl + 1) & (GS / 16 - 1)) == 0) {
                const int blk = (kc + kl) / (GS / 16);
                __local const float *swl = swl_all + (kl / (GS / 16)) * QOB;
#pragma unroll
                for (int i = 0; i < QTN; i++) {
                    const float s = sx[min(n0 + i, N - 1) * NB + blk];
#pragma unroll
                    for (int o = 0; o < QOB; o++) { f[i][o] = mad((float)acc[i][o], s * swl[o], f[i][o]); acc[i][o] = 0; }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < QTN; i++) {
        const int n = n0 + i;
        if (n >= nend) break;
#if EPI == 4
#pragma unroll
        for (int o = 0; o < QOB; o += 16) {
            float8 g = (float8)(f[i][o], f[i][o + 1], f[i][o + 2], f[i][o + 3], f[i][o + 4], f[i][o + 5], f[i][o + 6], f[i][o + 7]);
            float8 u = (float8)(f[i][o + 8], f[i][o + 9], f[i][o + 10], f[i][o + 11], f[i][o + 12], f[i][o + 13], f[i][o + 14], f[i][o + 15]);
#if CEN
            g += vload8(0, bias + o0 + o); u += vload8(0, bias + o0 + o + 8);
#endif
#if SITU   // MOSS: (4 tanh(g / 4) sigmoid(g)) * (25 tanh(u / 25))
            ST8(Y, n, (o0 + o) / 16, convert_half8(4.0f * tanh(g * 0.25f) / (1.0f + exp(-g)) * (25.0f * tanh(u * 0.04f))));
#else
            ST8(Y, n, (o0 + o) / 16, convert_half8(g / (1.0f + exp(-g)) * u));
#endif
        }
#elif EPI < 2
#pragma unroll
        for (int o = 0; o < QOB; o += 8) {
            float8 v = (float8)(f[i][o], f[i][o + 1], f[i][o + 2], f[i][o + 3], f[i][o + 4], f[i][o + 5], f[i][o + 6], f[i][o + 7]);
#if CEN
            v += vload8(0, bias + o0 + o);
#endif
#if EPI == 1 && GELU_TANH
            v = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
#elif EPI == 1
            v = 0.5f * v * (1.0f + erf(v * 0.70710678f));
#endif
            ST8(Y, n, (o0 + o) / 8, convert_half8(v));
        }
#else
#pragma unroll
        for (int o = 0; o < QOB; o += 4) {
            const int og = (o0 + o) / 4;
            float4 r = (float4)(f[i][o], f[i][o + 1], f[i][o + 2], f[i][o + 3]);
#if CEN
            r += vload4(0, bias + o0 + o);
#endif
#if EPI == 2
            res[og * N + n] += r * gate[og];
#else
            res[og * N + n] = r;
#endif
        }
#endif
    }
}

// ======== fused attention ========
// Self-attention without the score table: a work group of 128 items takes 64 queries of one head and walks the keys 64
// at a time - scores into local memory, a running max / sum per row (online softmax), the weighted values added into
// per-item outputs (4 queries x 16 dims). Reads the same Q / K / V layouts as the multiply path:
//   Q  (n, h*16 + d8)  8 dims of token n        K  (n/4, h*64 + d/2)  4 tokens x 2 dims
//   V  (h*32 + d/4, n/2)  4 dims x 2 tokens     O  (n, h*16 + d8)     like Q
// Launch: global (ceil(N/64)*128, heads), local (128, 1).
#ifdef KLOCAL
#define KLD(t, r) Kl[t][r]
#else
#define KLD(t, r) LD8(Ko, kt + (t) - sub * 2, h * 64 + (r))
#endif
#define QKJ(A, B, J) { const half8 k0 = KLD(sub * 2, c * 4 + J), k1 = KLD(sub * 2 + 1, c * 4 + J); \
    _Pragma("unroll") for (int i = 0; i < 4; i++) { \
        a0[i] = mad((half4)q[i].A, k0.lo, a0[i]); a0[i] = mad((half4)q[i].B, k0.hi, a0[i]); \
        a1[i] = mad((half4)q[i].A, k1.lo, a1[i]); a1[i] = mad((half4)q[i].B, k1.hi, a1[i]); } }
__kernel void flashattn(__read_only image2d_t Qo, __read_only image2d_t Ko, __read_only image2d_t Vo, int N,
                        __write_only image2d_t O) {
#ifdef PHALF
    __local half S[64][72];
#else
    __local float S[64][68];
#endif
    __local float red[64][8];
#ifdef KLOCAL
    __local half8 Kl[16][64];   // 64 keys (16 groups of 4) x 128 dims (64 pairs) of this head
#endif
    const int l = get_local_id(0), h = get_global_id(1), q0 = get_group_id(0) * 64;
    const int qg = l >> 3, sub = l & 7;
    // every item keeps the running max / sum of its own 4 queries (the 8 items of a query group hold the same copies)
    float m_run[4], l_run[4];
    float4 o[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        m_run[i] = -INFINITY; l_run[i] = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; d++) o[i][d] = 0;
    }
    for (int kb = 0; kb < N; kb += 64) {
        // scores: queries q0 + qg*4 + i, keys kb + sub*8 + 0..7
        float4 s0[4], s1[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { s0[i] = 0; s1[i] = 0; }
        const int kt = (kb + sub * 8) >> 2;
#ifdef KLOCAL
        for (int t = l; t < 16 * 64; t += 128) Kl[t >> 6][t & 63] = LD8(Ko, (kb >> 2) + (t >> 6), h * 64 + (t & 63));
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
#ifndef NOQK
        for (int c = 0; c < 16; c++) {
            half8 q[4];
#pragma unroll
            for (int i = 0; i < 4; i++) q[i] = LD8(Qo, q0 + qg * 4 + i, h * 16 + c);
            half4 a0[4], a1[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { a0[i] = 0; a1[i] = 0; }
            QKJ(s0, s1, 0) QKJ(s2, s3, 1) QKJ(s4, s5, 2) QKJ(s6, s7, 3)
#pragma unroll
            for (int i = 0; i < 4; i++) { s0[i] += convert_float4(a0[i]); s1[i] += convert_float4(a1[i]); }
        }
#endif
        const int kr = N - (kb + sub * 8);  // keys of this item still inside N
        if (kr < 8) {
            const float4 lo = (float4)(0, 1, 2, 3), hi = (float4)(4, 5, 6, 7);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s0[i] = select(s0[i], (float4)(-INFINITY), lo >= (float4)kr);
                s1[i] = select(s1[i], (float4)(-INFINITY), hi >= (float4)kr);
            }
        }
        // online softmax, spread over all items: partial maxima, then exponentials, then partial sums
        float al[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 m4 = fmax(s0[i], s1[i]);
            red[qg * 4 + i][sub] = fmax(fmax(m4.x, m4.y), fmax(m4.z, m4.w));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 ra = vload4(0, red[qg * 4 + i]), rb = vload4(1, red[qg * 4 + i]);
            const float4 r4 = fmax(ra, rb);
            const float mn = fmax(m_run[i], fmax(fmax(r4.x, r4.y), fmax(r4.z, r4.w)));
            al[i] = (m_run[i] == -INFINITY) ? 0.0f : exp(m_run[i] - mn);
            m_run[i] = mn;
            s0[i] = exp(s0[i] - mn); s1[i] = exp(s1[i] - mn);
#ifdef PHALF
            vstore_half4(s0[i], 0, &S[qg * 4 + i][sub * 8]); vstore_half4(s1[i], 1, &S[qg * 4 + i][sub * 8]);
#else
            vstore4(s0[i], 0, &S[qg * 4 + i][sub * 8]); vstore4(s1[i], 1, &S[qg * 4 + i][sub * 8]);
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // all of red read before the sums go in
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 t = s0[i] + s1[i];
            red[qg * 4 + i][sub] = t.x + t.y + t.z + t.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 t = vload4(0, red[qg * 4 + i]) + vload4(1, red[qg * 4 + i]);
            l_run[i] = l_run[i] * al[i] + (t.x + t.y + t.z + t.w);
#pragma unroll
            for (int d = 0; d < 4; d++) o[i][d] *= al[i];
        }
        // values: queries qg*4 + i, dims sub*16 + 0..15
#ifndef NOPV
#ifdef PVHALF
        for (int k8 = 0; k8 < 64; k8 += 8) {   // products in half, folded into float every 8 keys
            half4 acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int d = 0; d < 4; d++) acc[i][d] = 0;
#pragma unroll
            for (int kk = 0; kk < 8; kk += 2) {
                const int y = (kb + k8 + kk) >> 1;
                half8 v[4];
#pragma unroll
                for (int d = 0; d < 4; d++) v[d] = LD8(Vo, h * 32 + sub * 4 + d, y);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const half2 p = convert_half2(vload2(0, &S[qg * 4 + i][k8 + kk]));
#pragma unroll
                    for (int d = 0; d < 4; d++) acc[i][d] = mad((half4)p.x, v[d].lo, mad((half4)p.y, v[d].hi, acc[i][d]));
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int d = 0; d < 4; d++) o[i][d] += convert_float4(acc[i][d]);
        }
#else
        for (int kk = 0; kk < 64; kk += 2) {
            const int y = (kb + kk) >> 1;
            half8 v[4];
#pragma unroll
            for (int d = 0; d < 4; d++) v[d] = LD8(Vo, h * 32 + sub * 4 + d, y);
#pragma unroll
            for (int i = 0; i < 4; i++) {
#ifdef PHALF
                const float2 p = vload_half2(0, &S[qg * 4 + i][kk]);
#else
                const float2 p = vload2(0, &S[qg * 4 + i][kk]);
#endif
#pragma unroll
                for (int d = 0; d < 4; d++) o[i][d] = mad(p.x, convert_float4(v[d].lo), mad(p.y, convert_float4(v[d].hi), o[i][d]));
            }
        }
#endif
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int n = q0 + qg * 4 + i;
        if (n < N) {
            const float inv = 1.0f / l_run[i];
            ST8(O, n, h * 16 + sub * 2, convert_half8((float8)(o[i][0], o[i][1]) * inv));
            ST8(O, n, h * 16 + sub * 2 + 1, convert_half8((float8)(o[i][2], o[i][3]) * inv));
        }
    }
}

// ======== softmax inside the blending pass (2026-11-18)
// Self-attention without the separate softmax pass: P V where P = softmax(S) is formed on the fly from the raw
// scores S. Each work item keeps, per token, the running max m and running sum l of exp(s - m); when a new chunk of 8
// scores raises the max, the accumulators and l are scaled by exp(m_old - m_new). P never touches memory, which saves
// writing the N x N weight table and reading it back. Same layout and launch as gemml (EPI 0): X = S, W = V.
__kernel void gemmlsm(__read_only image2d_t X, __read_only image2d_t W, int N, int M, int K,
                      int xby, int wbx, int wby, int yby, int xb0, int wb0, int yb0, __write_only image2d_t Y,
                      int lvalid,     // keys from lvalid on are padding and get no weight
                      int causal) {   // 1: token n also ignores keys after n (prompt rows of Qwen-Image)
    __local half8 wl[KC / 8 * G * 4];
    const int nb = get_global_id(0), mb = get_global_id(1), b = get_global_id(2), lid = get_local_id(0);
    const int n0 = nb * TNI, xr = (b + xb0) * xby, wx = mb * G + (b + wb0) * wbx, wr = (b + wb0) * wby;
    float4 r[TNI][G];
    float m[TNI], l[TNI];
#pragma unroll
    for (int i = 0; i < TNI; i++) {
        m[i] = -INFINITY; l[i] = 0;
#pragma unroll
        for (int g = 0; g < G; g++) r[i][g] = 0;
    }
    for (int kc = 0; kc < K / 8; kc += KC / 8) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int idx = lid; idx < KC / 8 * G * 4; idx += WG) {
            const int k8l = idx / (G * 4), g = (idx / 4) % G, qq = idx & 3;
            wl[idx] = (kc + k8l < K / 8) ? LD8(W, wx + g, wr + (kc + k8l) * 4 + qq) : (half8)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int kn = min(KC / 8, K / 8 - kc);
        for (int k8l = 0; k8l < kn; k8l++) {
            half8 x[TNI];
#pragma unroll
            for (int i = 0; i < TNI; i++) {
                float8 s = convert_float8(LD8(X, n0 + i, xr + kc + k8l));
                const int key = (kc + k8l) * 8, lim = causal ? min(lvalid, n0 + i + 1) : lvalid;
                if (key + 8 > lim) {
                    const float8 kk = (float8)(0, 1, 2, 3, 4, 5, 6, 7) + (float)key;
                    s = select(s, (float8)(-INFINITY), kk >= (float8)lim);
                }
                const float cm = fmax(fmax(fmax(s.s0, s.s1), fmax(s.s2, s.s3)), fmax(fmax(s.s4, s.s5), fmax(s.s6, s.s7)));
                if (cm > m[i]) {
                    const float f = native_exp(m[i] - cm);   // 0 on the first chunk (m = -inf)
                    l[i] *= f;
#pragma unroll
                    for (int g = 0; g < G; g++) r[i][g] *= f;
                    m[i] = cm;
                }
                const float8 p = native_exp(s - m[i]);
                l[i] += dot(p.lo, (float4)1) + dot(p.hi, (float4)1);
                x[i] = XS(convert_half8(p));
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
                const int bi = (k8l * G + g) * 4;
                const half8 w0 = wl[bi], w1 = wl[bi + 1], w2 = wl[bi + 2], w3 = wl[bi + 3];
#pragma unroll
                for (int i = 0; i < TNI; i++) {
                    r[i][g] += DOT8(x[i], w0, w1, w2, w3);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < TNI; i++) {
        const int n = n0 + i;
        if (n >= N) break;
        const float inv = UNS / l[i];
#pragma unroll
        for (int g = 0; g < G; g += 2) {
            const float8 v = (float8)(r[i][g], r[i][g + 1]) * inv;
            ST8(Y, n, (b + yb0) * yby + (mb * G + g) / 2, convert_half8(v));
        }
    }
}
