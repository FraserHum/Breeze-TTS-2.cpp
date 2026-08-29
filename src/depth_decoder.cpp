#include "breeze/depth_decoder.h"
#include "breeze/sampling.h"

#include <algorithm>
#include <cmath>
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

// per step graph size: every tensor created in gctx (weights and KV live in their own
// contexts; they only borrow cgraph leaf slots, ~136 distinct per step):
//
//   per layer (35): rms_norm 2 · q/k/v (linear + reshape) 6 · rope 2 ·
//                   cache_append (view + cpy + view) 3 each, 6 · attention 9 ·
//                   attn_output linear 1 · add 1 · ffn rms_norm 2 · swiglu 5 · add 1
//                   (the 4 cache views are graph leaves, the other 31 are nodes)
//   fixed (9 at j=1, 8 otherwise): get_rows · concat (j=1 only) · in_proj ·
//                                  output rms_norm 2 · last (view + cont) 2 ·
//                                  head view · logits mul_mat (2 views, leaves)
//   inputs: 5 at j=1 (aud, h0, pos, ff, mask), 4 otherwise - all leaves
//
// = 429 / 428 non-input tensors + 5 / 4 inputs (434 / 432 total): ~379 compute nodes
// (cgraph->nodes) and up to ~54 gctx leaves (cgraph->leafs, excluding the ~136
// weight/KV leaves), so the worst case needs ~379 / ~190 array slots and 434 arena
// objects. All three are bounded by graph_cap (>= 1024): a 2x+ margin in every
// dimension. if depth ops are added or removed, re-derive this count; it is the only
// input to the graph_cap formula in init().
static size_t dd_step_n_nodes(int n_layer) {
    return 35ull * n_layer + 9;
}

// fail loudly on a model/cfg mismatch: the scratch buffers are sized from cfg, so a model
// that disagrees with cfg would silently overflow them otherwise
static void dd_require(ggml_tensor * t, const char * what, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (!t || t->ne[0] != ne0 || t->ne[1] != ne1 || (ne2 >= 0 && t->ne[2] != ne2)) {
        GGML_ABORT("depth_decoder: %s has shape [%lld, %lld, %lld], expected [%lld, %lld, %lld] - model/cfg mismatch\n",
                   what,
                   t ? (long long) t->ne[0] : -1, t ? (long long) t->ne[1] : -1, t ? (long long) t->ne[2] : -1,
                   (long long) ne0, (long long) ne1, ne2 >= 0 ? (long long) ne2 : -1);
    }
}

void DepthRunner::init(BreezeModel & m, int n_branches) {
    n_branch = n_branches;
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    const int hidden = m.cfg.hidden_size;
    const DepthConfig & c = m.cfg.dd;

    kv.init(m.backend, c.n_layer, c.head_dim, c.n_kv_head, nc + 1, n_branches);
    freq_factors = llama3_freq_factors(c);

    // per step scratch, sized from the model config - no size literals in run()/depth_step()
    logits_buf.assign((size_t) vs * n_branch, 0.0f);
    combined_logits.assign(vs, 0.0f);
    flat_hiddens.assign((size_t) hidden * n_branch, 0.0f);
    idx_staging.assign(n_branch, 0);
    pos_staging.assign(2 * n_branch, 0);
    // mask [total, n_tok] per step; the worst steps are j=1 (2*nb x 2*nb) and the last one
    // (nc*nb x nb)
    mask_staging.assign(std::max<size_t>((size_t) 4 * n_branch * n_branch,
                                         (size_t) nc * n_branch * n_branch), 0.0f);
    cpy_roots.reserve(2 * c.n_layer);

    // model/cfg consistency: depth_step reads the buffers at cfg sized extents, so a model
    // that disagrees with cfg must fail here, not overflow a buffer mid generation
    dd_require(m.w("dd.codebooks_head.weight"), "dd.codebooks_head.weight", c.hidden, vs, nc - 1);
    dd_require(m.w("audio_embd.weight"), "audio_embd.weight", hidden, (int64_t) nc * vs, -1);
    dd_require(m.w("dd.in_proj.weight"), "dd.in_proj.weight", c.hidden, hidden, -1);

    // deterministic node budget (see dd_step_n_nodes), doubled as margin for backend side
    // node expansion (SYCL/oneDNN) and model growth; the 1024 floor is llama.cpp's own
    // generic budget floor (graph_max_nodes), not a size observed after an overflow - v1
    // derived 256 (= n_layer*16+64), which sat below the real 429 node graph and aborted on
    // every backend, then hardcoded 1024 as a post-overflow fix with no derivation
    const size_t n_nodes = dd_step_n_nodes(c.n_layer);
    graph_cap = std::max<size_t>(1024, 2 * n_nodes);
    GGML_ASSERT(n_nodes < graph_cap);

    // host metadata arena, sized once (llama.cpp's buf_compute_meta formula): one
    // ggml_tensor_overhead() slot per tensor created in gctx - all of them are compute
    // nodes or graph leaves (inputs and views; see the budget comment above) - plus the
    // cgraph object itself. the bound is re-checked after every step build (depth_step
    // tripwire).
    const size_t mem = ggml_tensor_overhead() * graph_cap + ggml_graph_overhead_custom(graph_cap, false);
    graph_meta.assign(mem, 0);
    ggml_init_params p{ mem, graph_meta.data(), true }; // no_alloc: graph_meta owns the arena
    gctx = ggml_init(p);
    ggraph = nullptr; // (re)created per step after ggml_reset; points into graph_meta
}

void DepthRunner::free() {
    kv.free();
    if (gctx) ggml_free(gctx); // no_alloc ctx: frees the context struct only
    gctx = nullptr;
    ggraph = nullptr; // non-owning
    // release capacity, not just contents (clear() would keep ~0.6 MB alive)
    std::vector<uint8_t>().swap(graph_meta);
    std::vector<float>().swap(logits_buf);
    std::vector<float>().swap(combined_logits);
    std::vector<float>().swap(flat_hiddens);
    std::vector<float>().swap(freq_factors);
    std::vector<int32_t>().swap(idx_staging);
    std::vector<int32_t>().swap(pos_staging);
    std::vector<float>().swap(mask_staging);
    std::vector<ggml_tensor *>().swap(cpy_roots);
    graph_cap = 0;
}

// mirrors common.cpp's cache_append (which needs the per step Graph this optimization
// removes): store cur in the cache at pos and return a view over [0, pos+n); the cpy is
// recorded as an extra graph root (it is invisible from the logits dataflow and would
// otherwise be dropped)
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

// identical op sequence as main's dd_layer; the only differences are the plumbing (no per
// step Graph: cache_append's cpy is recorded in r.cpy_roots instead of Graph::write, and
// the persistent DepthRunner replaces the Graph & KVCache parameters)
static ggml_tensor * dd_layer(ggml_context * ctx, BreezeModel & m, DepthRunner & r,
                              ggml_tensor * x, int il, ggml_tensor * pos, ggml_tensor * ff,
                              ggml_tensor * mask, int start, int n) {
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

    ggml_tensor * kfull = dd_cache_append(ctx, r.cpy_roots, r.kv.k[il], k, start);
    ggml_tensor * vfull = dd_cache_append(ctx, r.cpy_roots, r.kv.v[il], v, start);
    ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    h = swiglu_ffn(ctx, h, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"), m.w(p + ".ffn_down.weight"));
    return ggml_add(ctx, res, h);
}

// runs one depth position for every CFG branch at once and writes the per branch logits
// straight into r.logits_buf, branch major so branch b starts at b * vocab (ggml_mul_mat
// output dims [a->ne[1], b->ne[1]], ggml.c:3285).
//
// no heap traffic: the graph is rebuilt into the reusable arena (ggml_reset, no
// ggml_init/ggml_free), inputs upload from the persistent staging vectors, and the
// readback lands directly in r.logits_buf.
static void depth_step(BreezeModel & m, DepthRunner & r, int start,
                       const std::vector<std::vector<float>> * hiddens,
                       int audio_code, int head_idx) {
    const DepthConfig & c = m.cfg.dd;
    const int nb = r.n_branch;
    const int n_pos = hiddens ? 2 : 1;
    const int n_tok = n_pos * nb;
    const int total = (start + n_pos) * nb;

    // rebase the reusable arena at offset 0 (churn free; the cgraph is recreated below and
    // points into the same buffer) and rebuild the topology - shapes vary per step (the KV
    // length in the mask grows with start), so a per step rebuild is required
    ggml_reset(r.gctx);
    r.cpy_roots.clear();
    r.ggraph = ggml_new_graph_custom(r.gctx, r.graph_cap, false);

    for (int b = 0; b < nb; b++) r.idx_staging[b] = audio_code;
    ggml_tensor * aud = ggml_new_tensor_2d(r.gctx, GGML_TYPE_I32, nb, 1);
    ggml_set_input(aud);
    ggml_tensor * embed = ggml_get_rows(r.gctx, m.w("audio_embd.weight"), aud); // [2048, nb]
    ggml_tensor * h0 = nullptr;
    if (hiddens) {
        GGML_ASSERT(hiddens->size() == (size_t) nb);
        for (int b = 0; b < nb; b++) {
            // guard: a branch with a mismatched hidden size would overflow flat_hiddens
            GGML_ASSERT((*hiddens)[b].size() == (size_t) m.cfg.hidden_size);
            std::memcpy(r.flat_hiddens.data() + (size_t) b * m.cfg.hidden_size,
                        (*hiddens)[b].data(), (size_t) m.cfg.hidden_size * sizeof(float));
        }
        h0 = ggml_new_tensor_3d(r.gctx, GGML_TYPE_F32, m.cfg.hidden_size, nb, 1);
        ggml_set_input(h0);
        embed = ggml_concat(r.gctx, h0, embed, 1); // [2048, 2*nb], position major
    }
    ggml_tensor * x = linear(r.gctx, m.w("dd.in_proj.weight"), embed); // [1024, n_tok]

    for (int i = 0; i < n_tok; i++) r.pos_staging[i] = start + i / nb;
    ggml_tensor * pos = ggml_new_tensor_2d(r.gctx, GGML_TYPE_I32, n_tok, 1);
    ggml_set_input(pos);
    ggml_tensor * ff = ggml_new_tensor_2d(r.gctx, GGML_TYPE_F32, (int) r.freq_factors.size(), 1);
    ggml_set_input(ff);

    // same values as build_branch_causal_mask, written into the persistent staging to avoid
    // the per step vector allocation
    const size_t mask_n = (size_t) total * n_tok;
    GGML_ASSERT(mask_n <= r.mask_staging.size()); // sized for the worst step in init
    for (int q = 0; q < n_tok; q++) {
        const int qb = q % nb;
        const int qpos = start + q / nb;
        for (int k = 0; k < total; k++)
            r.mask_staging[(size_t) q * total + k] = (k % nb == qb && k / nb <= qpos) ? 0.0f : -INFINITY;
    }
    ggml_tensor * mask = ggml_new_tensor_3d(r.gctx, GGML_TYPE_F32, total, n_tok, 1);
    ggml_set_input(mask);

    for (int il = 0; il < c.n_layer; il++)
        x = dd_layer(r.gctx, m, r, x, il, pos, ff, mask, start * nb, n_tok);
    x = rms_norm(r.gctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_cont(r.gctx, ggml_view_2d(r.gctx, x, c.hidden, nb, x->nb[1],
                                                        (size_t) (n_tok - nb) * x->nb[1]));
    ggml_tensor * head = m.w("dd.codebooks_head.weight");
    ggml_tensor * hw = ggml_view_2d(r.gctx, head, head->ne[0], head->ne[1], head->nb[1],
                                    (size_t) head_idx * head->nb[2]);
    ggml_tensor * logits = ggml_mul_mat(r.gctx, hw, last); // [vocab, nb]

    // same build order as Graph::compute: the cache-append cpy roots first (they are not
    // reachable from the logits dataflow and would be dropped otherwise), then the output
    ggml_set_output(logits);
    for (ggml_tensor * root : r.cpy_roots) ggml_build_forward_expand(r.ggraph, root);
    ggml_build_forward_expand(r.ggraph, logits);

    // budget tripwire (node side; the leaf side is bounded by the same cap and checked by
    // ggml's own unconditional overflow asserts - the cgraph struct is opaque outside
    // ggml's private headers, so no separate leaf check is possible in public API). if the
    // per step tensor count ever outgrows graph_cap, this names the cap instead of
    // aborting deep in ggml without context
    GGML_ASSERT(ggml_graph_n_nodes(r.ggraph) < (int) r.graph_cap);

    ggml_gallocr_alloc_graph(m.backend.alloc, r.ggraph);

    // inputs upload straight from the persistent staging (llama.cpp's set_inputs pattern),
    // after allocation so the data lands in the assigned buffer
    ggml_backend_tensor_set(aud, r.idx_staging.data(), 0, (size_t) nb * sizeof(int32_t));
    if (h0) ggml_backend_tensor_set(h0, r.flat_hiddens.data(), 0, (size_t) nb * m.cfg.hidden_size * sizeof(float));
    ggml_backend_tensor_set(pos, r.pos_staging.data(), 0, (size_t) n_tok * sizeof(int32_t));
    ggml_backend_tensor_set(ff, r.freq_factors.data(), 0, r.freq_factors.size() * sizeof(float));
    ggml_backend_tensor_set(mask, r.mask_staging.data(), 0, mask_n * sizeof(float));

    ggml_backend_graph_compute(m.backend.backend, r.ggraph);

    // readback size is derived from the tensor and checked against the pre allocated
    // buffer (a buffer driven copy would silently overflow, or leave a stale tail, on a
    // model/cfg mismatch)
    const size_t n_out = (size_t) logits->ne[0] * (size_t) logits->ne[1];
    GGML_ASSERT(n_out == r.logits_buf.size());
    ggml_backend_tensor_get(logits, r.logits_buf.data(), 0, n_out * sizeof(float));
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

    // no per step heap traffic: depth_step writes into logits_buf, the combine writes into
    // combined_logits, and sample_token is handed that persistent buffer
    std::vector<int> codes = { cb0 };
    codes.reserve(nc);
    for (int j = 1; j < nc; j++) {
        const int head_idx = j - 1;
        if (j == 1) depth_step(m, *this, 0, &hiddens, cb0, head_idx);
        else        depth_step(m, *this, j, nullptr, codes[head_idx] + head_idx * vs, head_idx);

        // logits_buf is branch major (branch b at b * vs, see the header); the expression
        // is main's CFG combine verbatim - cond is branch 0, uncond is branch 1
        if (n_branch > 1) {
            for (int i = 0; i < vs; i++)
                combined_logits[i] = logits_buf[vs + i] + cfg_scale * (logits_buf[i] - logits_buf[vs + i]);
        } else {
            std::memcpy(combined_logits.data(), logits_buf.data(), (size_t) vs * sizeof(float));
        }
        // sample_token destroys combined_logits in place, so it must never be fed a buffer
        // the loop did not just rewrite
        codes.push_back(j <= n_force ? force[j - 1] : sample_token(combined_logits, sp, rng));
    }
    return std::vector<int>(codes.begin() + 1, codes.end());
}

}
