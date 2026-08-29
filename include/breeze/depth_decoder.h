#pragma once

#include "breeze/model.h"
#include "breeze/sampling.h"

#include <random>
#include <vector>

namespace breeze {

// autoregressive residual decoder: predicts codebooks 1..num_codebooks-1 for one frame
struct DepthRunner {
    KVCache kv; // CFG branches share one cache, interleaved per position
    int n_branch = 1;
    std::vector<float> freq_factors;
    std::vector<float> logits_buf;
    std::vector<float> combined_logits;
    std::vector<float> flat_hiddens;

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
