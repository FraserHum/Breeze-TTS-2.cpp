#include "breeze/backbone.h"
#include "breeze/model.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kSlack = 8;
constexpr int kMaxContext = 4096;
constexpr int kMaxRepeats = 100000;

struct Options {
    std::string model;
    std::string backend = "cpu";
    int context = 1;
    int warmup = 1;
    int iterations = 5;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model PATH [--backend cpu|gpu] [--context N] "
        "[--warmup N] [--iterations N]\n", argv0);
}

static bool positive_int(const char * text, int max_value, int & value) {
    if (!text || !*text) return false;
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed < 1 || parsed > max_value) return false;
    value = static_cast<int>(parsed);
    return true;
}

static bool parse_args(int argc, char ** argv, Options & options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            options.model = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            options.backend = argv[++i];
        } else if (arg == "--cpu") {
            options.backend = "cpu";
        } else if (arg == "--gpu") {
            options.backend = "gpu";
        } else if (arg == "--context" && i + 1 < argc) {
            if (!positive_int(argv[++i], kMaxContext, options.context)) return false;
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!positive_int(argv[++i], kMaxRepeats, options.warmup)) return false;
        } else if (arg == "--iterations" && i + 1 < argc) {
            if (!positive_int(argv[++i], kMaxRepeats, options.iterations)) return false;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return false;
        } else {
            return false;
        }
    }
    return (options.backend == "cpu" || options.backend == "gpu") && !options.model.empty();
}

static uint32_t next_random(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill_embeddings(std::vector<float> & values, uint32_t seed) {
    uint32_t state = seed ? seed : 1;
    for (float & value : values) {
        const float unit = static_cast<float>(next_random(state) & 0x00ffffffu) / 16777216.0f;
        value = (unit * 2.0f - 1.0f) * 0.25f;
    }
}

static bool finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

static double elapsed_us(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

static bool exact_equal(const breeze::StepOut & a, const breeze::StepOut & b) {
    return a.hidden == b.hidden && a.logits == b.logits;
}

} // namespace

int main(int argc, char ** argv) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }
    breeze::BreezeModel model;
    if (!model.load(options.model, options.backend == "gpu")) {
        std::fprintf(stderr, "failed to load model: %s\n", options.model.c_str());
        return 3;
    }
    if ((options.backend == "gpu") != model.backend.is_gpu) {
        std::fprintf(stderr, "requested %s backend, got %s\n", options.backend.c_str(), model.backend.name());
        model.free();
        return 4;
    }
    const breeze::BackboneConfig & cfg = model.cfg.bb;
    if (cfg.hidden <= 0 || cfg.n_layer <= 0 || cfg.head_dim <= 0 || cfg.n_kv_head <= 0) {
        std::fprintf(stderr, "invalid backbone config in model\n");
        model.free();
        return 5;
    }

    const int hidden = cfg.hidden;
    const int prefix_tokens = options.context - 1;
    const size_t prefix_values = static_cast<size_t>(hidden) * static_cast<size_t>(prefix_tokens);
    std::vector<float> prefix(prefix_values);
    std::vector<float> token(static_cast<size_t>(hidden));
    fill_embeddings(prefix, 0x243f6a1du);
    fill_embeddings(token, 0x13198a2eu);

    breeze::BackboneState state;
    state.init(model, options.context + kSlack);
    double prefill_us = 0.0;
    if (prefix_tokens > 0) {
        const auto start = std::chrono::steady_clock::now();
        breeze::StepOut prefix_out = breeze::backbone_run(model, state, prefix, prefix_tokens);
        prefill_us = elapsed_us(start);
        if (!finite(prefix_out.hidden) || !finite(prefix_out.logits) || state.pos != prefix_tokens) {
            std::fprintf(stderr, "non-finite prefill output or unexpected prefix position\n");
            state.free();
            model.free();
            return 6;
        }
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.iterations));
    breeze::StepOut reference;
    bool all_finite = true;
    bool all_shapes = true;
    bool all_positions = true;
    bool exact_repeated = true;
    for (int i = 0; i < options.warmup + options.iterations; ++i) {
        state.pos = prefix_tokens;
        const auto start = std::chrono::steady_clock::now();
        breeze::StepOut out = breeze::backbone_run(model, state, token, 1);
        const double elapsed = elapsed_us(start);
        all_finite = all_finite && finite(out.hidden) && finite(out.logits);
        all_shapes = all_shapes && out.hidden.size() == static_cast<size_t>(cfg.hidden) &&
                     out.logits.size() == static_cast<size_t>(model.cfg.audio_vocab_size + 1);
        all_positions = all_positions && state.pos == options.context;
        if (i >= options.warmup) {
            if (samples.empty()) reference = out;
            else exact_repeated = exact_repeated && exact_equal(reference, out);
            samples.push_back(elapsed);
        }
    }

    const double total_us = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double mean_us = total_us / static_cast<double>(samples.size());
    const double min_us = *std::min_element(samples.begin(), samples.end());
    const double max_us = *std::max_element(samples.begin(), samples.end());
    std::printf("model=%s\nbackend=%s\ncontext=%d prefix_tokens=%d slack=%d warmup=%d iterations=%d\n",
                options.model.c_str(), model.backend.name(), options.context, prefix_tokens, kSlack,
                options.warmup, options.iterations);
    std::printf("config=hidden:%d layers:%d heads:%d kv_heads:%d head_dim:%d ffn:%d rope_theta:%.9g\n",
                cfg.hidden, cfg.n_layer, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.ffn, cfg.rope_theta);
    std::printf("timed_scope=full_host_backbone_run_including_graph_compute_and_output_readback\n");
    std::printf("prefill_us=%.3f mean_us=%.3f min_us=%.3f max_us=%.3f\n",
                prefill_us, mean_us, min_us, max_us);
    for (size_t i = 0; i < samples.size(); ++i)
        std::printf("sample[%zu]_us=%.3f\n", i, samples[i]);
    std::printf("finite_output=%s expected_output_shapes=%s expected_final_position=%s "
                "exact_repeated_hidden_and_logits=%s\n",
                all_finite ? "true" : "false", all_shapes ? "true" : "false",
                all_positions ? "true" : "false", exact_repeated ? "true" : "false");

    state.free();
    model.free();
    return all_finite && all_shapes && all_positions && exact_repeated ? 0 : 7;
}
