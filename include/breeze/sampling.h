#pragma once

#include <random>
#include <vector>

namespace breeze {

struct SampleParams {
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
};

// samples one token id from logits, applying suppression, repetition penalty, temperature, top-k, top-p
int sample_token(std::vector<float> & logits, const SampleParams & p, std::mt19937 & rng,
                 const std::vector<int> * history = nullptr,
                 const std::vector<int> * suppress = nullptr);

}
