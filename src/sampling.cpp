#include "breeze/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace breeze {

int sample_token(std::vector<float> & logits, const SampleParams & p, std::mt19937 & rng,
                 const std::vector<int> * history, const std::vector<int> * suppress) {
    const int n = (int) logits.size();

    if (suppress) {
        for (int t : *suppress)
            if (t >= 0 && t < n) logits[t] = -std::numeric_limits<float>::infinity();
    }

    if (history && p.repetition_penalty != 1.0f) {
        for (int t : *history) {
            if (t < 0 || t >= n) continue;
            logits[t] = logits[t] > 0 ? logits[t] / p.repetition_penalty : logits[t] * p.repetition_penalty;
        }
    }

    const float temp = p.temperature > 0 ? p.temperature : 1.0f;

    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;

    int keep = (p.top_k > 0 && p.top_k < n) ? p.top_k : n;
    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(keep);

    float maxl = logits[idx[0]];
    std::vector<float> probs(keep);
    float sum = 0.0f;
    for (int i = 0; i < keep; i++) {
        probs[i] = std::exp((logits[idx[i]] - maxl) / temp);
        sum += probs[i];
    }
    for (float & v : probs) v /= sum;

    if (p.top_p < 1.0f) {
        float cum = 0.0f;
        int cut = keep;
        for (int i = 0; i < keep; i++) {
            cum += probs[i];
            if (cum >= p.top_p) { cut = i + 1; break; }
        }
        probs.resize(cut);
        idx.resize(cut);
        float s = 0.0f;
        for (float v : probs) s += v;
        for (float & v : probs) v /= s;
    }

    if (probs.size() == 1) return idx[0]; // degenerate: no draw, keeps caller rng streams aligned

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return idx[dist(rng)];
}

}
