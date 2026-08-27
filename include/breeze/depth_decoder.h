#pragma once

#include "breeze/model.h"

#include <random>
#include <vector>

namespace breeze {

// autoregressive residual decoder: predicts codebooks 1..num_codebooks-1 for one frame
struct DepthRunner {
    std::vector<KVCache> kv; // one per CFG branch
    std::vector<float> freq_factors;

    void init(BreezeModel & m, int n_branches);
    void free();

    // hiddens holds the backbone last hidden per branch (cond first, then uncond); returns cb1..cb_{n-1}
    std::vector<int> run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                         int cb0, float cfg_scale, std::mt19937 & rng);
};

}
