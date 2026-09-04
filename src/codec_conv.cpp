#include "breeze/codec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace breeze {
namespace codec_detail {

static bool env_flag1(const char * name) {
    const char * e = std::getenv(name);
    return e && std::strcmp(e, "1") == 0;
}

ggml_tensor * conv1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int dilation) {
    const int K = (int) w->ne[0];
    int pad = (K - 1) * dilation - (stride - 1);
    if (pad < 0) pad = 0;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t n_out = (x->ne[0] + pad - (int64_t) dilation * (K - 1) - 1) / stride + 1;
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    if (y->ne[0] != n_out) {
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, n_out, y->ne[1], y->nb[1], 0));
    }
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

ggml_tensor * convtr1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                              ggml_tensor * x, int stride) {
    if (!env_flag1("BREEZE_VOC_CONVT_MATMUL")) {
        ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
        const int keep = (int) x->ne[0] * stride;
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, keep, y->ne[1], y->nb[1], 0));
        if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
        return y;
    }
    // BREEZE_VOC_CONVT_MATMUL=1: decompose the 1-D transposed conv into per-tap batched
    // matmuls plus the dedicated col2im_1d scatter. Mathematically identical to
    // ggml_conv_transpose_1d (p0=0, d0=1): the reference kernel scatters
    //     out[t, co] = sum_{l, k: t = l*stride + k} sum_ci w[k, co, ci] * x[l, ci]
    // and col2im_1d gathers exactly
    //     out[t, co] = sum_{l, k: t = l*stride + k} M[co*K + k, l]
    // so M[co*K + k, l] = (w[k] @ x)[co, l] reproduces it. The Vulkan conv_transpose_1d
    // kernel is barrier-bound on the 780M (86 ms/op, ~5-10% of peak); this route uses the
    // matmul and col2im pipelines. Note: on coopmat2 devices the matmul runs with f16
    // converted operands, so results are not bit-exact with the f32 reference.
    const int64_t K = w->ne[0];      // kernel taps
    const int64_t Cout = w->ne[1];   // output channels
    const int64_t Cin = w->ne[2];    // input channels
    const int64_t L = x->ne[0];      // input length

    // the kernel arrives as [K, Cout, Cin] but the underlying buffer is [Cin, Cout, K]
    // (K fastest, Cin slowest): w->nb = [4, 4*Cin, 4*Cin*Cout] expressed as [K,Cout,Cin]
    // gives K stride 4, Cout stride 8, Cin stride 8192. A zero-copy view keeps Cin
    // slowest (transposed -> mul_mat assert), so we physically copy it into a contiguous
    // [Cin, Cout, K] (Cin fastest, K slowest = clean batched-matmul layout, batch stride
    // = Cin*Cout*4). The single cont+permute does the full transpose in one pass.
    ggml_tensor * wt = ggml_cont(ctx, ggml_permute(ctx, w, 2, 1, 0, 3)); // [Cin, Cout, K], Cin fastest, K batch
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [Cin, L], contiguous
    ggml_tensor * xb = ggml_repeat_4d(ctx, xt, Cin, L, K, 1);            // [Cin, L, K]
    if (std::getenv("BREEZE_VOC_CONVT_DBG")) {
        fprintf(stderr, "[convt_dbg] w  type=%d ne=[%ld %ld %ld %ld] nb=[%zu %zu %zu %zu]\n",
                (int) w->type, (long)w->ne[0],(long)w->ne[1],(long)w->ne[2],(long)w->ne[3],
                w->nb[0],w->nb[1],w->nb[2],w->nb[3]);
        fprintf(stderr, "[convt_dbg] wt type=%d ne=[%ld %ld %ld %ld] nb=[%zu %zu %zu %zu] transposed=%d\n",
                (int) wt->type, (long)wt->ne[0],(long)wt->ne[1],(long)wt->ne[2],(long)wt->ne[3],
                wt->nb[0],wt->nb[1],wt->nb[2],wt->nb[3], (int)ggml_is_transposed(wt));
        fprintf(stderr, "[convt_dbg] xb type=%d ne=[%ld %ld %ld %ld] nb=[%zu %zu %zu %zu] transposed=%d\n",
                (int) xb->type, (long)xb->ne[0],(long)xb->ne[1],(long)xb->ne[2],(long)xb->ne[3],
                xb->nb[0],xb->nb[1],xb->nb[2],xb->nb[3], (int)ggml_is_transposed(xb));
    }
    ggml_tensor * y3 = ggml_mul_mat(ctx, wt, xb);                        // [Cout, L, K]
    ggml_tensor * m3 = ggml_permute(ctx, y3, 0, 2, 1, 3);                // [Cout, K, L]
    ggml_tensor * m2 = ggml_reshape_2d(ctx, ggml_cont(ctx, m3), Cout * K, L);
    ggml_tensor * y = ggml_col2im_1d(ctx, m2, stride, (int) Cout, 0);    // [L*stride + K - 1, Cout]

    const int keep = (int) x->ne[0] * stride;
    y = ggml_cont(ctx, ggml_view_2d(ctx, y, keep, y->ne[1], y->nb[1], 0));
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

ggml_tensor * resnet_block(ggml_context * ctx, BreezeModel & m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * h = ggml_elu(ctx, x);
    h = conv1d_causal(ctx, m.w(p + ".block.1.conv.weight"), m.w(p + ".block.1.conv.bias"), h, 1, 1);
    h = ggml_elu(ctx, h);
    h = conv1d_causal(ctx, m.w(p + ".block.3.conv.weight"), m.w(p + ".block.3.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

// depthwise conv as shift and multiply, ggml conv ops have no group support
// x is [L, C], w is [C, K]
ggml_tensor * depthwise1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                 ggml_tensor * x, int K) {
    const int64_t L = x->ne[0];
    const int64_t C = x->ne[1];
    ggml_tensor * xp = ggml_pad_ext(ctx, x, 0, K - 1, 0, 0, 0, 0, 0, 0);
    xp = ggml_roll(ctx, xp, K - 1, 0, 0, 0);
    ggml_tensor * acc = nullptr;
    for (int k = 0; k < K; k++) {
        ggml_tensor * seg = ggml_cont(ctx, ggml_view_2d(ctx, xp, L, C, xp->nb[1], (size_t) k * xp->nb[0]));
        ggml_tensor * wk = ggml_cont(ctx, ggml_transpose(ctx,
            ggml_view_2d(ctx, w, C, 1, w->nb[1], (size_t) k * w->nb[1])));
        ggml_tensor * t = ggml_mul(ctx, seg, wk);
        acc = acc ? ggml_add(ctx, acc, t) : t;
    }
    if (b) acc = ggml_add(ctx, acc, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return acc;
}

ggml_tensor * seanet_encoder(ggml_context * ctx, BreezeModel & m, ggml_tensor * x) {
    const std::vector<int> & up = m.cfg.codec.upsampling_ratios;
    auto conv = [&](int i, ggml_tensor * in, int s) {
        const std::string p = "codec.enc." + std::to_string(i);
        return conv1d_causal(ctx, m.w(p + ".conv.weight"), m.w(p + ".conv.bias"), in, s, 1);
    };
    const int conv_idx[4] = { 3, 6, 9, 12 };
    const int res_idx[4] = { 1, 4, 7, 10 };
    const int ratios[4] = { up[3], up[2], up[1], up[0] };
    ggml_tensor * h = conv(0, x, 1);
    for (int s = 0; s < 4; s++) {
        h = resnet_block(ctx, m, "codec.enc." + std::to_string(res_idx[s]), h);
        h = ggml_elu(ctx, h);
        h = conv(conv_idx[s], h, ratios[s]);
    }
    h = ggml_elu(ctx, h);
    return conv(14, h, 1);
}

}
}
