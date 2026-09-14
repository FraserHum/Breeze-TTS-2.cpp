#include "breeze/common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    bool cpu = false, mask_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu") cpu = true;
        else if (std::string(argv[i]) == "--mask-only") mask_only = true;
        else return 2;
    }
    breeze::Backend backend;
    backend.init(!cpu);
    if (!cpu && !backend.is_gpu) return 2;
    bool pass = true;
    for (int branches : {1, 2}) for (int positions : {2, 3, 4, 15, 16}) {
        const int hd = 128, heads = 8, kv_heads = 2;
        const int queries = (positions == 2 ? 2 : 1) * branches;
        const int tokens = positions * branches, capacity = 17 * branches;
        auto values = [](int n, float phase) {
            std::vector<float> result(n);
            for (int i = 0; i < n; ++i) result[i] = std::sin(i * 0.137f + phase);
            return result;
        };
        breeze::Graph graph;
        auto q_values = values(hd * heads * queries, 0.1f);
        if (mask_only) std::fill(q_values.begin(), q_values.end(), 0.0f);
        auto * q = graph.input_f32(q_values, hd, heads, queries);
        auto * k_full = graph.input_f32(values(hd * kv_heads * capacity, 0.2f), hd, kv_heads, capacity);
        auto v_values = values(hd * kv_heads * capacity, 0.3f);
        if (mask_only) for (int t = 0; t < capacity; ++t) for (int h = 0; h < kv_heads; ++h) {
            const float value = t >= tokens ? 8192.0f : 4.0f * h + 32.0f * (t % branches)
                + (t / branches == positions - 1 ? 64.0f : 0.0f);
            std::fill_n(v_values.begin() + (t * kv_heads + h) * hd, hd, value);
        }
        auto * v_full = graph.input_f32(v_values, hd, kv_heads, capacity);
        auto * k = ggml_view_3d(graph.ctx, k_full, hd, kv_heads, tokens, k_full->nb[1], k_full->nb[2], 0);
        auto * v = ggml_view_3d(graph.ctx, v_full, hd, kv_heads, tokens, v_full->nb[1], v_full->nb[2], 0);
        auto mask_values = breeze::build_branch_causal_mask(queries, tokens,
            positions - queries / branches, branches);
        auto * mask = graph.input_f32(mask_values, tokens, queries);
        auto * regular = breeze::attention(graph.ctx, q, k, v, mask, 1.0f / std::sqrt(float(hd)), heads, kv_heads);
        auto * flash = breeze::attention_flash(graph.ctx, q, k, v,
            ggml_cast(graph.ctx, mask, GGML_TYPE_F16), 1.0f / std::sqrt(float(hd)));
        pass = pass && regular->ne[0] == hd * heads && regular->ne[1] == queries &&
            flash->ne[0] == hd * heads && flash->ne[1] == queries;
        graph.mark_output(regular);
        graph.compute(backend, flash);
        auto a = breeze::tensor_to_f32(regular), b = breeze::tensor_to_f32(flash);
        const bool shape_ok = a.size() == b.size() && a.size() == size_t(hd * heads * queries);
        bool ok = shape_ok;
        double analytic_max = 0;
        if (mask_only && ok) for (int t = 0; t < queries; ++t) for (int h = 0; h < heads; ++h) {
            const int position = positions - queries / branches + t / branches;
            const float expected = 4.0f * (h / (heads / kv_heads)) + 32.0f * (t % branches)
                + (position == positions - 1 ? 64.0f / positions : 0.0f);
            for (int d = 0; d < hd; ++d) {
                const size_t i = size_t(t * heads + h) * hd + d;
                analytic_max = std::max(analytic_max, double(std::max(std::abs(a[i] - expected), std::abs(b[i] - expected))));
                ok = ok && std::isfinite(a[i]) && std::isfinite(b[i]) && analytic_max < 0.001;
            }
        }
        double max_abs = 0, error = 0, energy = 0;
        for (size_t i = 0; shape_ok && i < a.size(); ++i) {
            ok = ok && std::isfinite(a[i]) && std::isfinite(b[i]);
            const double d = double(a[i]) - b[i];
            max_abs = std::max(max_abs, std::abs(d));
            error += d * d;
            energy += double(a[i]) * a[i];
        }
        const double relative = std::sqrt(error / std::max(energy, 1e-30));
        ok = ok && max_abs < 0.001 && relative < 0.001;
        std::printf("mode=%s branches=%d positions=%d max_abs=%.9g relative_l2=%.9g analytic_max=%.9g %s\n",
            mask_only ? "mask" : "numeric", branches, positions, max_abs, relative, analytic_max, ok ? "PASS" : "FAIL");
        pass = pass && ok;
    }
    backend.free();
    return pass ? 0 : 1;
}
