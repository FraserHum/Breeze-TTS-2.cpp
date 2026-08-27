#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// keeps embeddings at Q8_0 minimum and leaves norms, biases, codebooks and conv kernels untouched
static ggml_type pick_type(const std::string & name, const ggml_tensor * t, ggml_type req) {
    if (t->type != GGML_TYPE_F16) return t->type;
    if (ggml_n_dims(t) != 2) return GGML_TYPE_F16;
    const int64_t k = t->ne[0];
    const bool is_embed = name == "te.token_embd.weight" || name == "audio_embd.weight";
    if (is_embed) return (k % 32 == 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    if (req == GGML_TYPE_Q4_K || req == GGML_TYPE_Q6_K) {
        if (k % 256 == 0) return req;
        if (k % 32 == 0) return GGML_TYPE_Q8_0;
        return GGML_TYPE_F16;
    }
    if (req == GGML_TYPE_Q8_0) return (k % 32 == 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    return GGML_TYPE_F16;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        printf("usage: breeze-quantize <in-f16.gguf> <out.gguf> <q8_0|q6_k|q4_k>\n");
        return 1;
    }
    std::string inp = argv[1], outp = argv[2], q = argv[3];
    ggml_type req;
    if (q == "q8_0" || q == "Q8_0") req = GGML_TYPE_Q8_0;
    else if (q == "q6_k" || q == "Q6_K") req = GGML_TYPE_Q6_K;
    else if (q == "q4_k" || q == "Q4_K") req = GGML_TYPE_Q4_K;
    else { fprintf(stderr, "unknown quant type: %s\n", q.c_str()); return 1; }

    ggml_context * ctx = nullptr;
    gguf_init_params p{ false, &ctx };
    gguf_context * in = gguf_init_from_file(inp.c_str(), p);
    if (!in) { fprintf(stderr, "failed to open %s\n", inp.c_str()); return 1; }

    const int64_t nt = gguf_get_n_tensors(in);
    std::vector<ggml_type> types(nt);
    size_t total = 0;
    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        types[i] = pick_type(name, t, req);
        total += ggml_row_size(types[i], t->ne[0]) * ggml_nrows(t);
    }

    ggml_init_params op{ total + (size_t) nt * ggml_tensor_overhead() + (16u << 20), nullptr, false };
    ggml_context * octx = ggml_init(op);
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);

    std::vector<float> f32;
    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        ggml_tensor * src = ggml_get_tensor(ctx, name);
        ggml_type tt = types[i];
        ggml_tensor * dst = ggml_new_tensor(octx, tt, ggml_n_dims(src), src->ne);
        ggml_set_name(dst, name);
        if (tt == src->type) {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        } else {
            const int64_t n = ggml_nelements(src);
            f32.resize(n);
            if (src->type == GGML_TYPE_F16) ggml_fp16_to_fp32_row((const ggml_fp16_t *) src->data, f32.data(), n);
            else memcpy(f32.data(), src->data, n * sizeof(float));
            ggml_quantize_chunk(tt, f32.data(), dst->data, 0, ggml_nrows(src), src->ne[0], nullptr);
        }
        gguf_add_tensor(out, dst);
        printf("\r%lld/%lld %-40s -> %s", (long long) (i + 1), (long long) nt, name, ggml_type_name(tt));
        fflush(stdout);
    }
    printf("\nwriting %s ...\n", outp.c_str());
    if (!gguf_write_to_file(out, outp.c_str(), false)) { fprintf(stderr, "write failed\n"); return 1; }
    gguf_free(out);
    gguf_free(in);
    ggml_free(octx);
    ggml_free(ctx);
    printf("done\n");
    return 0;
}
