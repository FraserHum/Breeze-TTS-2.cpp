#pragma once

#include "breeze/model.h"
#include "breeze/sampling.h"

#include <cstdint>
#include <random>
#include <vector>

namespace breeze {

// one static compute graph for one depth step: built once in init with that step's
// shapes (KV length, append offset, head slice, and the step 1 CFG concat) baked in,
// re-executed per step by updating only the input leaves' data
struct DepthStepGraph {
    std::vector<uint8_t> arena;     // no-alloc ggml context memory
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> cpy_roots;
    ggml_tensor * aud = nullptr;    // [nb] i32 audio code, same for every branch
    ggml_tensor * h0 = nullptr;     // [hidden, nb] f32, step 1 only (cond then uncond)
    ggml_tensor * pos = nullptr;    // [n_tok] i32 positions
    ggml_tensor * ff = nullptr;     // rope freq factors, constant
    ggml_tensor * mask = nullptr;   // [total, n_tok] f32 branch-causal
    ggml_tensor * logits = nullptr; // [vs, nb] f32 output, branch major
};

// autoregressive residual decoder: predicts codebooks 1..num_codebooks-1 for one frame
struct DepthRunner {
    KVCache kv; // CFG branches share one cache, interleaved per position
    int n_branch = 1;
    std::vector<float> freq_factors;
    std::vector<float> logits_buf;
    std::vector<float> combined_logits;
    std::vector<float> flat_hiddens;

    std::vector<float> logits_buf;
    std::vector<float> combined_logits;
    std::vector<float> flat_hiddens;

    std::vector<DepthStepGraph> graphs; // one per step, graphs[j-1] is step j
    ggml_gallocr_t depth_alloc = nullptr; // dedicated: backbone/codec churn on m.backend.alloc would dangle these pointers
    size_t graph_cap = 0;

    std::vector<int32_t> idx_staging;
    std::vector<int32_t> pos_staging;
    std::vector<float> mask_staging;

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

// opt-in per-phase timing of the last DepthRunner::run() call, armed by BREEZE_DEPTH_STEP_TIMING=1.
// numerically inert: steady_clock reads and storage only, no graph ops, no RNG draws
struct RtDepthTiming {
    double stage_ms = 0.0, set_ms = 0.0, comp_ms = 0.0, d2h_ms = 0.0, sample_ms = 0.0;
};
bool rt_depth_timing_enabled();
const RtDepthTiming & rt_depth_last();

}
