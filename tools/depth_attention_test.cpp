#include "breeze/common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--cpu")) return 2;
    breeze::Backend backend;
    backend.init(argc == 1);
    if (argc == 1 && !backend.is_gpu) return 2;
    bool pass = true;
    for (int branches : {1, 2}) for (int positions : {2, 4, 16}) {
        const int hd = 128, heads = 8, kv_heads = 2;
        const int queries = (positions == 2 ? 2 : 1) * branches;
        const int tokens = positions * branches, capacity = 17 * branches;
        auto values = [](int n, float phase) {
            std::vector<float> result(n);
            for (int i = 0; i < n; ++i) result[i] = std::sin(i * 0.137f + phase);
            return result;
        };
        breeze::Graph graph;
        auto * q = graph.input_f32(values(hd * heads * queries, 0.1f), hd, heads, queries);
        auto * k_full = graph.input_f32(values(hd * kv_heads * capacity, 0.2f), hd, kv_heads, capacity);
        auto * v_full = graph.input_f32(values(hd * kv_heads * capacity, 0.3f), hd, kv_heads, capacity);
        auto * k = ggml_view_3d(graph.ctx, k_full, hd, kv_heads, tokens, k_full->nb[1], k_full->nb[2], 0);
        auto * v = ggml_view_3d(graph.ctx, v_full, hd, kv_heads, tokens, v_full->nb[1], v_full->nb[2], 0);
        auto mask_values = breeze::build_branch_causal_mask(queries, tokens,
            positions - queries / branches, branches);
        auto * mask = graph.input_f32(mask_values, tokens, queries);
        auto * regular = breeze::attention(graph.ctx, q, k, v, mask, 1.0f / std::sqrt(float(hd)), heads, kv_heads);
        auto * flash = breeze::attention_flash(graph.ctx, q, k, v,
            ggml_cast(graph.ctx, mask, GGML_TYPE_F16), 1.0f / std::sqrt(float(hd)));
        graph.mark_output(regular);
        graph.compute(backend, flash);
        auto a = breeze::tensor_to_f32(regular), b = breeze::tensor_to_f32(flash);
        bool ok = a.size() == b.size() && a.size() == size_t(hd * heads * queries);
        double max_abs = 0, error = 0, energy = 0;
        for (size_t i = 0; ok && i < a.size(); ++i) {
            ok = std::isfinite(a[i]) && std::isfinite(b[i]);
            const double d = double(a[i]) - b[i];
            max_abs = std::max(max_abs, std::abs(d));
            error += d * d;
            energy += double(a[i]) * a[i];
        }
        const double relative = std::sqrt(error / std::max(energy, 1e-30));
        ok = ok && max_abs < 0.001 && relative < 0.001;
        std::printf("branches=%d positions=%d max_abs=%.9g relative_l2=%.9g %s\n",
            branches, positions, max_abs, relative, ok ? "PASS" : "FAIL");
        pass = pass && ok;
    }
    backend.free();
    return pass ? 0 : 1;
}
