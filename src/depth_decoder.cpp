#include "breeze/depth_decoder.h"
#include "breeze/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace breeze {

static std::vector<float> llama3_freq_factors(const DepthConfig & c) {
    const int half = c.head_dim / 2;
    std::vector<float> ff(half, 1.0f);
    const float pi = 3.14159265358979323846f;
    const float low_wl = c.rope_orig_ctx / c.rope_low_freq;
    const float high_wl = c.rope_orig_ctx / c.rope_high_freq;
    for (int i = 0; i < half; i++) {
        float freq = std::pow(c.rope_theta, -2.0f * i / c.head_dim);
        float wavelen = 2.0f * pi / freq;
        if (wavelen > low_wl) {
            ff[i] = c.rope_factor;
        } else if (wavelen < high_wl) {
            ff[i] = 1.0f;
        } else {
            float smooth = (c.rope_orig_ctx / wavelen - c.rope_low_freq) / (c.rope_high_freq - c.rope_low_freq);
            ff[i] = 1.0f / ((1.0f - smooth) / c.rope_factor + smooth);
        }
    }
    return ff;
}

static size_t dd_step_n_nodes(int n_layer) {
    return 35ull * n_layer + 9;
}

static void dd_require(ggml_tensor * t, const char * what, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (!t || t->ne[0] != ne0 || t->ne[1] != ne1 || (ne2 >= 0 && t->ne[2] != ne2)) {
        GGML_ABORT("depth_decoder: %s has shape [%lld, %lld, %lld], expected [%lld, %lld, %lld] - model/cfg mismatch\n",
                   what,
                   t ? (long long) t->ne[0] : -1, t ? (long long) t->ne[1] : -1, t ? (long long) t->ne[2] : -1,
                   (long long) ne0, (long long) ne1, ne2 >= 0 ? (long long) ne2 : -1);
    }
}

static DepthStepGraph build_step(BreezeModel & m, const KVCache & kv, int nb, int j, size_t graph_cap);

void DepthRunner::init(BreezeModel & m, int n_branches) {
    n_branch = n_branches;
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    const int hidden = m.cfg.hidden_size;
    const DepthConfig & c = m.cfg.dd;

    kv.init(m.backend, c.n_layer, c.head_dim, c.n_kv_head, nc + 1, n_branches);
    freq_factors = llama3_freq_factors(c);

    logits_buf.assign((size_t) vs * n_branch, 0.0f);
    combined_logits.assign(vs, 0.0f);
    flat_hiddens.assign((size_t) hidden * n_branch, 0.0f);
    idx_staging.assign(n_branch, 0);
    pos_staging.assign(2 * n_branch, 0);
    mask_staging.assign(std::max<size_t>((size_t) 4 * n_branch * n_branch,
                                         (size_t) nc * n_branch * n_branch), 0.0f);

    dd_require(m.w("dd.codebooks_head.weight"), "dd.codebooks_head.weight", c.hidden, vs, nc - 1);
    dd_require(m.w("audio_embd.weight"), "audio_embd.weight", hidden, (int64_t) nc * vs, -1);
    dd_require(m.w("dd.in_proj.weight"), "dd.in_proj.weight", hidden, c.hidden, -1);

    const size_t n_nodes = dd_step_n_nodes(c.n_layer);
    graph_cap = std::max<size_t>(1024, 2 * n_nodes);
    GGML_ASSERT(n_nodes < graph_cap);

    // each step has its own baked shapes, so every step gets a static graph; the graphs
    // share one dedicated allocator because generate() interleaves backbone and codec
    // computes on m.backend.alloc, and a vbuffer growth there frees and dangles the
    // tensor addresses of any graph allocated on the shared one
    depth_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend.backend));
    const int n_step = nc - 1;
    graphs.resize(n_step);
    for (int j = 1; j <= n_step; j++)
        graphs[j - 1] = build_step(m, kv, n_branch, j, graph_cap);

    // reserve the vbuffer exactly once: the largest graph (step 1) first, then assert
    // no later graph grows it, since growth frees and reallocates the vbuffer and
    // dangles every address recorded for the graphs already allocated
    ggml_gallocr_alloc_graph(depth_alloc, graphs[0].graph);
    const size_t vbuf = ggml_gallocr_get_buffer_size(depth_alloc, 0);
    for (int j = 1; j < n_step; j++) {
        ggml_gallocr_alloc_graph(depth_alloc, graphs[j].graph);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(depth_alloc, 0) == vbuf);
    }
    fprintf(stderr, "depth decoder: %d static graphs, compute buffer %zu bytes\n", n_step, vbuf);
}

void DepthRunner::free() {
    kv.free();
    if (depth_alloc) ggml_gallocr_free(depth_alloc);
    depth_alloc = nullptr;
    for (DepthStepGraph & g : graphs) {
        if (g.ctx) ggml_free(g.ctx); // no-alloc: frees the context struct only, the arena is ours
        g.ctx = nullptr;
        g.graph = nullptr;
        std::vector<uint8_t>().swap(g.arena);
        std::vector<ggml_tensor *>().swap(g.cpy_roots);
        g.aud = g.h0 = g.pos = g.ff = g.mask = g.logits = nullptr;
    }
    std::vector<DepthStepGraph>().swap(graphs);
    std::vector<float>().swap(logits_buf);
    std::vector<float>().swap(combined_logits);
    std::vector<float>().swap(flat_hiddens);
    std::vector<float>().swap(freq_factors);
    std::vector<int32_t>().swap(idx_staging);
    std::vector<int32_t>().swap(pos_staging);
    std::vector<float>().swap(mask_staging);
    graph_cap = 0;
}

static ggml_tensor * dd_cache_append(ggml_context * ctx, std::vector<ggml_tensor *> & roots,
                                     ggml_tensor * cache, ggml_tensor * cur, int pos) {
    const int hd = (int) cache->ne[0];
    const int nkv = (int) cache->ne[1];
    const int n = (int) cur->ne[2];
    ggml_tensor * dst = ggml_view_3d(ctx, cache, hd, nkv, n, cache->nb[1], cache->nb[2],
                                     (size_t) pos * cache->nb[2]);
    roots.push_back(ggml_cpy(ctx, cur, dst));
    return ggml_view_3d(ctx, cache, hd, nkv, pos + n, cache->nb[1], cache->nb[2], 0);
}

static ggml_tensor * dd_layer(ggml_context * ctx, BreezeModel & m, ggml_tensor * x, int il,
                              ggml_tensor * pos, ggml_tensor * ff, ggml_tensor * mask,
                              int start, int n,
                              const std::vector<ggml_tensor *> & k_cache,
                              const std::vector<ggml_tensor *> & v_cache,
                              std::vector<ggml_tensor *> & cpy_roots) {
    const DepthConfig & c = m.cfg.dd;
    const std::string p = "dd.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), h), c.head_dim, c.n_head, n);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), h), c.head_dim, c.n_kv_head, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), h), c.head_dim, c.n_kv_head, n);
    q = ggml_rope_ext(ctx, q, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * kfull = dd_cache_append(ctx, cpy_roots, k_cache[il], k, start);
    ggml_tensor * vfull = dd_cache_append(ctx, cpy_roots, v_cache[il], v, start);
    ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    h = swiglu_ffn(ctx, h, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"), m.w(p + ".ffn_down.weight"));
    return ggml_add(ctx, res, h);
}

// builds the static compute graph for depth step j (1..num_codebooks-1) the way v2's
// depth_step built it per step, with the step j shapes baked in: the KV length and
// append offset, the head-weight slice, and the CFG concat (present only for j=1)
// are not writable after build, so every step gets its own graph
static DepthStepGraph build_step(BreezeModel & m, const KVCache & kv, int nb, int j, size_t graph_cap) {
    const DepthConfig & c = m.cfg.dd;
    const bool has_hidden = j == 1;
    const int start = has_hidden ? 0 : j;
    const int n_pos = has_hidden ? 2 : 1;
    const int n_tok = n_pos * nb;
    const int total = (start + n_pos) * nb;
    const int head_idx = j - 1;

    // per-step node budget: 35/layer + 9 fixed (j=1, with the CFG concat) / 8 (j>=2)
    const size_t n_nodes = 35ull * c.n_layer + (has_hidden ? 9 : 8);
    const size_t mem = ggml_tensor_overhead() * (n_nodes + 5) + ggml_graph_overhead_custom(graph_cap, false);
    DepthStepGraph g;
    g.arena.assign(mem, 0);
    ggml_init_params p{ mem, g.arena.data(), true };
    g.ctx = ggml_init(p);

    // persistent input leaves, updated per step with ggml_backend_tensor_set
    g.aud = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, nb, 1);
    ggml_set_input(g.aud);
    ggml_tensor * embed = ggml_get_rows(g.ctx, m.w("audio_embd.weight"), g.aud); // [2048, nb]
    if (has_hidden) {
        g.h0 = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, m.cfg.hidden_size, nb, 1);
        ggml_set_input(g.h0);
        embed = ggml_concat(g.ctx, g.h0, embed, 1); // [2048, 2*nb], position major
    }
    ggml_tensor * x = linear(g.ctx, m.w("dd.in_proj.weight"), embed); // [1024, n_tok]

    g.pos = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_tok, 1);
    ggml_set_input(g.pos);
    g.ff = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, c.head_dim / 2, 1);
    ggml_set_input(g.ff);
    g.mask = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, total, n_tok, 1);
    ggml_set_input(g.mask);

    for (int il = 0; il < c.n_layer; il++)
        x = dd_layer(g.ctx, m, x, il, g.pos, g.ff, g.mask, start * nb, n_tok, kv.k, kv.v, g.cpy_roots);
    x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_cont(g.ctx, ggml_view_2d(g.ctx, x, c.hidden, nb, x->nb[1],
                                                       (size_t) (n_tok - nb) * x->nb[1]));
    ggml_tensor * head = m.w("dd.codebooks_head.weight");
    ggml_tensor * hw = ggml_view_2d(g.ctx, head, head->ne[0], head->ne[1], head->nb[1],
                                    (size_t) head_idx * head->nb[2]);
    g.logits = ggml_mul_mat(g.ctx, hw, last); // [vocab, nb]

    ggml_set_output(g.logits);
    g.graph = ggml_new_graph_custom(g.ctx, graph_cap, false);
    for (ggml_tensor * root : g.cpy_roots) ggml_build_forward_expand(g.graph, root);
    ggml_build_forward_expand(g.graph, g.logits);

    GGML_ASSERT(ggml_graph_n_nodes(g.graph) < (int) graph_cap);
    // no-alloc arena: the tensors and the cgraph are the only objects, so usage must
    // fit the derived budget above; overflow would abort inside ggml_init's allocator
    GGML_ASSERT(ggml_used_mem(g.ctx) <= mem);
    return g;
}

std::vector<int> DepthRunner::run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                                  int cb0, float cfg_scale, std::mt19937 & rng,
                                  const SampleParams * sp_in, const int * force, int n_force) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    kv.reset();

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;
    if (sp_in) sp = *sp_in;

    std::vector<int> codes = { cb0 };
    codes.reserve(nc);
    for (int j = 1; j < nc; j++) {
        const int head_idx = j - 1;
        DepthStepGraph & g = graphs[j - 1];
        const int start = j == 1 ? 0 : j;
        const int n_tok = (int) g.pos->ne[0];
        const int total = (int) g.mask->ne[0];

        if (j == 1) {
            GGML_ASSERT(hiddens.size() == (size_t) n_branch);
            for (int b = 0; b < n_branch; b++) {
                GGML_ASSERT(hiddens[b].size() == (size_t) m.cfg.hidden_size);
                std::memcpy(flat_hiddens.data() + (size_t) b * m.cfg.hidden_size,
                            hiddens[b].data(), (size_t) m.cfg.hidden_size * sizeof(float));
            }
        }
        const int audio_code = j == 1 ? cb0 : codes[head_idx] + head_idx * vs;
        for (int b = 0; b < n_branch; b++) idx_staging[b] = audio_code;
        for (int i = 0; i < n_tok; i++) pos_staging[i] = start + i / n_branch;

        const size_t mask_n = (size_t) total * n_tok;
        GGML_ASSERT(mask_n <= mask_staging.size());
        for (int q = 0; q < n_tok; q++) {
            const int qb = q % n_branch;
            const int qpos = start + q / n_branch;
            for (int k = 0; k < total; k++)
                mask_staging[(size_t) q * total + k] = (k % n_branch == qb && k / n_branch <= qpos) ? 0.0f : -INFINITY;
        }

        // the graph is already allocated on depth_alloc; per step only the input data moves
        ggml_backend_tensor_set(g.aud, idx_staging.data(), 0, (size_t) n_branch * sizeof(int32_t));
        if (g.h0) ggml_backend_tensor_set(g.h0, flat_hiddens.data(), 0, (size_t) n_branch * m.cfg.hidden_size * sizeof(float));
        ggml_backend_tensor_set(g.pos, pos_staging.data(), 0, (size_t) n_tok * sizeof(int32_t));
        ggml_backend_tensor_set(g.ff, freq_factors.data(), 0, freq_factors.size() * sizeof(float));
        ggml_backend_tensor_set(g.mask, mask_staging.data(), 0, mask_n * sizeof(float));

        ggml_backend_graph_compute(m.backend.backend, g.graph);

        const size_t n_out = (size_t) g.logits->ne[0] * (size_t) g.logits->ne[1];
        GGML_ASSERT(n_out == logits_buf.size());
        ggml_backend_tensor_get(g.logits, logits_buf.data(), 0, n_out * sizeof(float));

        if (n_branch > 1) {
            for (int i = 0; i < vs; i++)
                combined_logits[i] = logits_buf[vs + i] + cfg_scale * (logits_buf[i] - logits_buf[vs + i]);
        } else {
            std::memcpy(combined_logits.data(), logits_buf.data(), (size_t) vs * sizeof(float));
        }
        codes.push_back(j <= n_force ? force[j - 1] : sample_token(combined_logits, sp, rng));
    }
    return std::vector<int>(codes.begin() + 1, codes.end());
}

}
