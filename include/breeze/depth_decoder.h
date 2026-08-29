#pragma once

#include "breeze/model.h"
#include "breeze/sampling.h"

#include <cstdint>
#include <random>
#include <vector>

namespace breeze {

// autoregressive residual decoder: predicts codebooks 1..num_codebooks-1 for one frame
struct DepthRunner {
    KVCache kv; // CFG branches share one cache, interleaved per position
    int n_branch = 1;
    std::vector<float> freq_factors;

    // per step scratch, sized once from the model config in init() and released in free().
    // the run() loop does no heap allocation: every per step value is written through these
    // buffers (or the staging vectors below), which outlive the whole generation.
    //
    // logits_buf is branch major: branch b's [audio_vocab_size] block starts at
    // b * audio_vocab_size (ggml_mul_mat output dims [a->ne[1], b->ne[1]], vocab fastest,
    // ggml.c:3285). depth_step() writes into it and run()'s CFG combine reads it in exactly
    // this layout; keep the two in sync.
    //
    // combined_logits is fully rewritten (CFG combine or copy) before every sample_token()
    // call; sample_token() mutates the buffer in place (suppression, penalties, temperature
    // destroy it), so no path may feed it a buffer the loop did not just rewrite.
    std::vector<float> logits_buf;      // [audio_vocab_size * n_branch]
    std::vector<float> combined_logits; // [audio_vocab_size]
    std::vector<float> flat_hiddens;    // [hidden_size * n_branch], cond branch first

    // reusable depth step graph, following llama.cpp's buf_compute_meta pattern
    // (src/llama-graph.cpp, llm_graph_result::reset): the host metadata arena is sized once
    // in init() from the node budget and bound with no_alloc, so each step is ggml_reset
    // (rebase the object arena at offset 0) + rebuild into the same buffer - no per step
    // ggml_init/ggml_free churn. the topology is rebuilt per step because the KV length in
    // the causal mask grows with the step.
    std::vector<uint8_t> graph_meta;
    ggml_context * gctx = nullptr;
    ggml_cgraph * ggraph = nullptr; // non-owning; points into graph_meta, recreated per step
    size_t graph_cap = 0;

    // persistent input staging (llama.cpp's set_inputs pattern): the input tensors are
    // recreated in gctx each step, but their host data lives here, sized once in init() and
    // uploaded with ggml_backend_tensor_set - no per step staging vectors.
    std::vector<int32_t> idx_staging;  // [n_branch]
    std::vector<int32_t> pos_staging;  // [2 * n_branch]
    std::vector<float> mask_staging;   // [max over steps of total * n_tok]
    std::vector<ggml_tensor *> cpy_roots; // the 2*n_layer cache-append cpy ops, cleared per step

    void init(BreezeModel & m, int n_branches);
    void free();

    // hiddens holds the backbone last hidden per branch (cond first, then uncond); returns cb1..cb_{n-1}.
    // sp overrides the model's own sampling settings, null uses them.
    // force supplies the first n_force codebooks instead of sampling them
    std::vector<int> run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                         int cb0, float cfg_scale, std::mt19937 & rng,
                         const SampleParams * sp = nullptr,
                         const int * force = nullptr, int n_force = 0);
};

}
