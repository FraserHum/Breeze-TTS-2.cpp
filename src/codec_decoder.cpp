#include "breeze/codec.h"

#include <cmath>
#include <string>
#include <vector>

namespace breeze {
namespace codec_detail {

// snake beta activation: x + sin(x*alpha)^2 / beta, both params stored in log space
static ggml_tensor * snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * la, ggml_tensor * lb) {
    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, la->ne[0]);
    ggml_tensor * beta = ggml_reshape_2d(ctx, ggml_exp(ctx, lb), 1, lb->ne[0]);
    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    return ggml_add(ctx, x, ggml_div(ctx, ggml_sqr(ctx, s), beta));
}

// convnext block over [L, C]
static ggml_tensor * convnext(ggml_context * ctx, BreezeModel & m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * dw = m.w(p + ".dw.weight");
    ggml_tensor * h = depthwise1d_causal(ctx, dw, m.w(p + ".dw.bias"), x, (int) dw->ne[1]);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = layer_norm(ctx, h, m.w(p + ".norm.weight"), m.w(p + ".norm.bias"), 1e-6f);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw1.weight"), h), m.w(p + ".pw1.bias"));
    h = ggml_gelu_erf(ctx, h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw2.weight"), h), m.w(p + ".pw2.bias"));
    h = ggml_mul(ctx, h, m.w(p + ".gamma"));
    return ggml_add(ctx, x, ggml_cont(ctx, ggml_transpose(ctx, h)));
}

static ggml_tensor * residual_unit(ggml_context * ctx, BreezeModel & m, const std::string & p,
                                   ggml_tensor * x, int dilation) {
    ggml_tensor * h = snake_beta(ctx, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = conv1d_causal(ctx, m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, 1, dilation);
    h = snake_beta(ctx, h, m.w(p + ".a2"), m.w(p + ".b2"));
    h = conv1d_causal(ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

// rebuild the continuous latent from the residual codebooks, first stage split from the rest
static ggml_tensor * quantizer_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                                      const std::vector<int> & codes, int n_cb, int T) {
    auto lookup = [&](const std::string & name, int cb) {
        std::vector<int32_t> idx(T);
        for (int t = 0; t < T; t++) idx[t] = codes[(size_t) t * n_cb + cb];
        ggml_tensor * ids = g.input_i32(idx, T);
        return ggml_get_rows(ctx, m.w(name), ids);
    };

    ggml_tensor * first = lookup("codec.dq.first.0.embed", 0);
    first = linear(ctx, m.w("codec.dq.first.out_proj.weight"), first);

    ggml_tensor * rest = nullptr;
    for (int cb = 1; cb < n_cb; cb++) {
        ggml_tensor * e = lookup("codec.dq.rest." + std::to_string(cb - 1) + ".embed", cb);
        rest = rest ? ggml_add(ctx, rest, e) : e;
    }
    if (rest) {
        rest = linear(ctx, m.w("codec.dq.rest.out_proj.weight"), rest);
        first = ggml_add(ctx, first, rest);
    }
    return ggml_cont(ctx, ggml_transpose(ctx, first));
}

ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                             const std::vector<int> & codes, int n_cb, int T) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T);
    h = conv1d_causal(ctx, m.w("codec.dpre.conv.weight"), m.w("codec.dpre.conv.bias"), h, 1, 1);

    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = vocoder_transformer(ctx, m, g, h, T);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));

    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsampling_ratios[i]);
        h = convnext(ctx, m, p, h);
    }

    h = conv1d_causal(ctx, m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1, 1);
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(ctx, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsample_rates[i]);
        for (int j = 0; j < 3; j++) {
            h = residual_unit(ctx, m, p + ".res." + std::to_string(j), h, dilations[j]);
        }
    }

    h = snake_beta(ctx, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = conv1d_causal(ctx, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1, 1);
    return ggml_clamp(ctx, h, -1.0f, 1.0f);
}

}
}
