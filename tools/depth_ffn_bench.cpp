#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#if BREEZE_FFN_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kLayers = 12;
constexpr int kHidden = 1024;
constexpr int kFfn = 8192;

enum WeightKind {
    Gate = 0,
    Up = 1,
    Down = 2,
};

struct Options {
    std::string backend = "gpu";
    std::string model;
    std::string json;
    int layers = kLayers;
    int warmup = 2;
    int iterations = 8;
};

struct ModelWeights {
    gguf_context * gguf = nullptr;
    ggml_context * metadata_ctx = nullptr;
    ggml_context * tensor_ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::array<std::array<ggml_tensor *, 3>, kLayers> weights{};

    ModelWeights() = default;

    ~ModelWeights() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (tensor_ctx) ggml_free(tensor_ctx);
        if (metadata_ctx) ggml_free(metadata_ctx);
        if (gguf) gguf_free(gguf);
    }

    ModelWeights(const ModelWeights &) = delete;
    ModelWeights & operator=(const ModelWeights &) = delete;
};

struct Chain {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * silu = nullptr;
    ggml_tensor * product = nullptr;
    ggml_tensor * down = nullptr;
    ggml_cgraph * full_graph = nullptr;
    std::array<ggml_cgraph *, 5> split_graphs{};
};

struct FfnCase {
    int n = 0;
    ggml_tensor * input = nullptr;
    std::array<Chain, kLayers> normal{};
    std::array<Chain, kLayers> fused{};
    ggml_cgraph * normal_batch = nullptr;
    ggml_cgraph * fused_batch = nullptr;
};

struct OutputStats {
    uint64_t hash = 0;
    std::vector<float> values;
};

struct TimedResult {
    double mean_us = 0.0;
    int submissions = 0;
    OutputStats output;
    uint64_t repeat_hash = 0;
};

struct CompareStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
};

struct CaseResults {
    std::array<TimedResult, 5> values{};
    CompareStats normal_vs_swiglu;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [--backend gpu|cpu] [--model PATH] [--json PATH] "
        "[--layers 1..12] [--warmup N] [--iterations N]\n", argv0);
}

static bool parse_int(const char * text, int & value) {
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > 1000000) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static bool parse_args(int argc, char ** argv, Options & options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            options.backend = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            options.model = argv[++i];
        } else if (arg == "--json" && i + 1 < argc) {
            options.json = argv[++i];
        } else if (arg == "--layers" && i + 1 < argc) {
            if (!parse_int(argv[++i], options.layers)) return false;
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!parse_int(argv[++i], options.warmup)) return false;
        } else if (arg == "--iterations" && i + 1 < argc) {
            if (!parse_int(argv[++i], options.iterations)) return false;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return false;
        } else {
            return false;
        }
    }
    return (options.backend == "gpu" || options.backend == "cpu")
        && options.layers >= 1 && options.layers <= kLayers
        && options.iterations >= 1;
}

static std::string default_model_path() {
    if (const char * env = std::getenv("BREEZE_FFN_MODEL")) {
        if (*env) return env;
    }
    const char * home = std::getenv("HOME");
    if (home) {
        const std::string local = std::string(home) + "/.cache/breeze-models/breeze-tts-2-q4_k.gguf";
        if (FILE * f = std::fopen(local.c_str(), "rb")) {
            std::fclose(f);
            return local;
        }
    }
    return "/models/breeze-tts-2-q4_k.gguf";
}

static uint32_t next_random(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill_normalized(std::vector<float> & values, uint32_t seed) {
    uint32_t state = seed ? seed : 1;
    for (float & value : values) {
        const float unit = static_cast<float>(next_random(state) & 0x00ffffffu) / 16777216.0f;
        value = (unit * 2.0f - 1.0f) * 0.25f;
    }
}

static uint64_t fnv1a(const void * data, size_t size, uint64_t hash = 1469598103934665603ull) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static ggml_context * make_context(size_t bytes) {
    ggml_init_params params = { bytes, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for %zu bytes\n", bytes);
        std::exit(2);
    }
    return ctx;
}

static void require_status(ggml_status status, const char * what) {
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s failed: %s\n", what, ggml_status_to_string(status));
        std::exit(2);
    }
}

static bool read_at(FILE * file, uint64_t offset, void * data, size_t size) {
#if defined(_WIN32)
    if (_fseeki64(file, static_cast<__int64>(offset), SEEK_SET) != 0) return false;
#else
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) return false;
#endif
    return std::fread(data, 1, size, file) == size;
}

static const char * weight_suffix(int kind) {
    switch (kind) {
    case Gate: return "ffn_gate.weight";
    case Up: return "ffn_up.weight";
    default: return "ffn_down.weight";
    }
}

static bool load_weights(const std::string & path, ggml_backend_t backend, ModelWeights & model) {
    gguf_init_params params = { true, &model.metadata_ctx };
    model.gguf = gguf_init_from_file(path.c_str(), params);
    if (!model.gguf) {
        std::fprintf(stderr, "cannot read GGUF metadata: %s\n", path.c_str());
        return false;
    }

    const size_t tensor_bytes = ggml_tensor_overhead() * kLayers * 3 + 4096;
    ggml_context * weight_ctx = make_context(tensor_bytes);
    std::array<std::array<size_t, 3>, kLayers> sizes{};
    for (int layer = 0; layer < kLayers; ++layer) {
        for (int kind = Gate; kind <= Down; ++kind) {
            const std::string name = "dd.blk." + std::to_string(layer) + "." + weight_suffix(kind);
            const int64_t id = gguf_find_tensor(model.gguf, name.c_str());
            if (id < 0) {
                std::fprintf(stderr, "missing model tensor: %s\n", name.c_str());
                ggml_free(weight_ctx);
                return false;
            }
            const int64_t * ne = gguf_get_tensor_ne(model.gguf, id);
            const ggml_type type = gguf_get_tensor_type(model.gguf, id);
            const int expected_k = kind == Down ? kFfn : kHidden;
            const int expected_m = kind == Down ? kHidden : kFfn;
            if (type != GGML_TYPE_Q4_K || ne[0] != expected_k || ne[1] != expected_m
                    || ne[2] != 1 || ne[3] != 1) {
                std::fprintf(stderr, "unexpected %s: type=%s shape=[%" PRId64 ",%" PRId64 "]\n",
                             name.c_str(), ggml_type_name(type), ne[0], ne[1]);
                ggml_free(weight_ctx);
                return false;
            }
            ggml_tensor * tensor = ggml_new_tensor_2d(weight_ctx, type, ne[0], ne[1]);
            model.weights[layer][kind] = tensor;
            sizes[layer][kind] = gguf_get_tensor_size(model.gguf, id);
            if (sizes[layer][kind] != ggml_nbytes(tensor)) {
                std::fprintf(stderr, "size mismatch for %s: file=%zu tensor=%zu\n",
                             name.c_str(), sizes[layer][kind], ggml_nbytes(tensor));
                ggml_free(weight_ctx);
                return false;
            }
        }
    }

    model.tensor_ctx = weight_ctx;
    model.buffer = ggml_backend_alloc_ctx_tensors(model.tensor_ctx, backend);
    if (!model.buffer) {
        std::fprintf(stderr, "model weight allocation failed\n");
        return false;
    }
    ggml_backend_buffer_set_usage(model.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open GGUF payload: %s\n", path.c_str());
        return false;
    }
    const uint64_t data_offset = gguf_get_data_offset(model.gguf);
    std::vector<uint8_t> bytes;
    for (int layer = 0; layer < kLayers; ++layer) {
        for (int kind = Gate; kind <= Down; ++kind) {
            const std::string name = "dd.blk." + std::to_string(layer) + "." + weight_suffix(kind);
            const int64_t id = gguf_find_tensor(model.gguf, name.c_str());
            bytes.resize(sizes[layer][kind]);
            if (!read_at(file, data_offset + gguf_get_tensor_offset(model.gguf, id),
                         bytes.data(), bytes.size())) {
                std::fprintf(stderr, "cannot read GGUF payload: %s\n", name.c_str());
                std::fclose(file);
                return false;
            }
            ggml_backend_tensor_set(model.weights[layer][kind], bytes.data(), 0, bytes.size());
        }
    }
    std::fclose(file);
    return true;
}

static ggml_cgraph * make_single_graph(ggml_context * ctx, ggml_tensor * node) {
    ggml_set_output(node);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_graph_add_node(graph, node);
    return graph;
}

static ggml_cgraph * make_full_graph(ggml_context * ctx, ggml_tensor * output) {
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, output);
    return graph;
}

static ggml_cgraph * make_batch_graph(ggml_context * ctx,
                                      const std::array<ggml_tensor *, kLayers> & outputs,
                                      int layers) {
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, layers * 8, false);
    for (int layer = 0; layer < layers; ++layer) {
        ggml_set_output(outputs[layer]);
        ggml_build_forward_expand(graph, outputs[layer]);
    }
    return graph;
}

static Chain build_chain(ggml_context * ctx, const ModelWeights & model, int layer,
                         ggml_tensor * input, bool use_swiglu) {
    Chain chain;
    chain.gate = ggml_mul_mat(ctx, model.weights[layer][Gate], input);
    chain.up = ggml_mul_mat(ctx, model.weights[layer][Up], input);
    if (use_swiglu) {
        chain.product = ggml_swiglu_split(ctx, chain.gate, chain.up);
    } else {
        chain.silu = ggml_silu(ctx, chain.gate);
        chain.product = ggml_mul(ctx, chain.silu, chain.up);
    }
    chain.down = ggml_mul_mat(ctx, model.weights[layer][Down], chain.product);
    chain.full_graph = make_full_graph(ctx, chain.down);
    chain.split_graphs[0] = make_single_graph(ctx, chain.gate);
    chain.split_graphs[1] = make_single_graph(ctx, chain.up);
    chain.split_graphs[2] = chain.silu ? make_single_graph(ctx, chain.silu) : nullptr;
    chain.split_graphs[3] = make_single_graph(ctx, chain.product);
    chain.split_graphs[4] = make_single_graph(ctx, chain.down);
    return chain;
}

static FfnCase build_case(ggml_context * ctx, const ModelWeights & model, int n, int layers) {
    FfnCase result;
    result.n = n;
    result.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, n);
    ggml_set_input(result.input);
    for (int layer = 0; layer < kLayers; ++layer) {
        result.normal[layer] = build_chain(ctx, model, layer, result.input, false);
        result.fused[layer] = build_chain(ctx, model, layer, result.input, true);
    }
    std::array<ggml_tensor *, kLayers> normal_outputs{};
    std::array<ggml_tensor *, kLayers> fused_outputs{};
    for (int layer = 0; layer < layers; ++layer) {
        normal_outputs[layer] = result.normal[layer].down;
        fused_outputs[layer] = result.fused[layer].down;
    }
    result.normal_batch = make_batch_graph(ctx, normal_outputs, layers);
    result.fused_batch = make_batch_graph(ctx, fused_outputs, layers);
    return result;
}

static void compute_and_sync(ggml_backend_t backend, ggml_cgraph * graph) {
    require_status(ggml_backend_graph_compute(backend, graph), "graph compute");
    ggml_backend_synchronize(backend);
}

enum class Variant {
    Split,
    PerFfn,
    Batched,
    PerFfnSwiglu,
    BatchedSwiglu,
};

static const char * variant_name(Variant variant) {
    switch (variant) {
    case Variant::Split: return "split";
    case Variant::PerFfn: return "per_ffn";
    case Variant::Batched: return "batched";
    case Variant::PerFfnSwiglu: return "per_ffn_swiglu";
    default: return "batched_swiglu";
    }
}

static int submission_count(Variant variant, int layers) {
    return variant == Variant::Split ? layers * 5
        : variant == Variant::Batched || variant == Variant::BatchedSwiglu ? 1 : layers;
}

static void run_once(ggml_backend_t backend, const FfnCase & ffn, Variant variant, int layers) {
    const bool fused = variant == Variant::PerFfnSwiglu || variant == Variant::BatchedSwiglu;
    if (variant == Variant::Batched || variant == Variant::BatchedSwiglu) {
        compute_and_sync(backend, fused ? ffn.fused_batch : ffn.normal_batch);
        return;
    }
    for (int layer = 0; layer < layers; ++layer) {
        const Chain & chain = fused ? ffn.fused[layer] : ffn.normal[layer];
        if (variant == Variant::Split) {
            for (ggml_cgraph * graph : chain.split_graphs) {
                if (graph) compute_and_sync(backend, graph);
            }
        } else {
            compute_and_sync(backend, chain.full_graph);
        }
    }
}

static OutputStats read_outputs(ggml_backend_t backend, const FfnCase & ffn,
                                Variant variant, int layers) {
    (void) backend;
    const bool fused = variant == Variant::PerFfnSwiglu || variant == Variant::BatchedSwiglu;
    OutputStats stats;
    for (int layer = 0; layer < layers; ++layer) {
        const ggml_tensor * tensor = fused ? ffn.fused[layer].down : ffn.normal[layer].down;
        const size_t count = static_cast<size_t>(ggml_nelements(tensor));
        const size_t old_size = stats.values.size();
        stats.values.resize(old_size + count);
        ggml_backend_tensor_get(tensor, stats.values.data() + old_size, 0, count * sizeof(float));
    }
    for (float value : stats.values) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "non-finite real-weight FFN output\n");
            std::exit(6);
        }
    }
    stats.hash = fnv1a(stats.values.data(), stats.values.size() * sizeof(float));
    return stats;
}

static TimedResult run_variant(ggml_backend_t backend, const FfnCase & ffn, Variant variant,
                               int layers, int warmup, int iterations) {
    for (int pass = 0; pass < warmup; ++pass) run_once(backend, ffn, variant, layers);
    const auto start = std::chrono::steady_clock::now();
    for (int pass = 0; pass < iterations; ++pass) run_once(backend, ffn, variant, layers);
    const auto finish = std::chrono::steady_clock::now();

    TimedResult result;
    result.mean_us = std::chrono::duration<double, std::micro>(finish - start).count()
        / static_cast<double>(iterations);
    result.submissions = submission_count(variant, layers);
    result.output = read_outputs(backend, ffn, variant, layers);

    run_once(backend, ffn, variant, layers);
    result.repeat_hash = read_outputs(backend, ffn, variant, layers).hash;
    if (result.output.hash != result.repeat_hash) {
        std::fprintf(stderr, "%s n=%d produced unstable output hash\n",
                     variant_name(variant), ffn.n);
        std::exit(7);
    }
    return result;
}

static void print_result(const FfnCase & ffn, Variant variant, const TimedResult & result,
                         int layers) {
    std::printf("variant=%s n=%d layers=%d submissions=%d mean_us=%.3f per_submission_us=%.3f "
                "checksum=%016" PRIx64 " repeat_checksum=%016" PRIx64 "\n",
                variant_name(variant), ffn.n, layers, result.submissions, result.mean_us,
                result.mean_us / result.submissions, result.output.hash, result.repeat_hash);
}

static CompareStats compare_results(const TimedResult & normal, const TimedResult & fused) {
    CompareStats result;
    for (size_t i = 0; i < normal.output.values.size(); ++i) {
        const double a = normal.output.values[i];
        const double b = fused.output.values[i];
        result.max_abs = std::max(result.max_abs, std::abs(a - b));
        result.max_rel = std::max(result.max_rel, std::abs(a - b) / std::max(1.0, std::abs(a)));
    }
    return result;
}

static void print_tolerance(const FfnCase & ffn, const TimedResult & normal,
                            const TimedResult & fused, const CompareStats & compare) {
    std::printf("compare=normal_vs_swiglu n=%d max_abs=%.9g max_rel=%.9g "
                "normal_checksum=%016" PRIx64 " swiglu_checksum=%016" PRIx64 "\n",
                ffn.n, compare.max_abs, compare.max_rel, normal.output.hash, fused.output.hash);
}

static std::string json_quote(const std::string & value) {
    std::string result = "\"";
    for (const char c : value) {
        if (c == '\\' || c == '\"') result += '\\';
        result += c;
    }
    result += '"';
    return result;
}

static std::string hex_hash(uint64_t value) {
    char buffer[17];
    std::snprintf(buffer, sizeof(buffer), "%016" PRIx64, value);
    return buffer;
}

static void write_json(const Options & options, const std::string & device,
                       const std::array<CaseResults, 2> & cases) {
    if (options.json.empty()) return;
    std::ofstream output(options.json);
    if (!output) {
        std::fprintf(stderr, "cannot write JSON receipt: %s\n", options.json.c_str());
        std::exit(8);
    }
    output << "{\n"
           << "  \"backend\": " << json_quote(options.backend) << ",\n"
           << "  \"device\": " << json_quote(device) << ",\n"
           << "  \"model\": " << json_quote(options.model) << ",\n"
           << "  \"layers\": " << options.layers << ",\n"
           << "  \"warmup\": " << options.warmup << ",\n"
           << "  \"iterations\": " << options.iterations << ",\n"
           << "  \"weights\": \"real_model_Q4_K\",\n"
           << "  \"input\": \"synthetic_normalized_f32_scale_0.25\",\n"
           << "  \"timed_scope\": \"graph_compute_plus_synchronize\",\n"
           << "  \"cases\": [\n";
    output << std::setprecision(12);
    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        if (case_index) output << ",\n";
        output << "    {\"n\": " << case_index + 1 << ", \"variants\": {\n";
        for (size_t variant_index = 0; variant_index < cases[case_index].values.size(); ++variant_index) {
            if (variant_index) output << ",\n";
            const TimedResult & result = cases[case_index].values[variant_index];
            output << "      " << json_quote(variant_name(static_cast<Variant>(variant_index)))
                   << ": {\"mean_us\": " << result.mean_us
                   << ", \"submissions\": " << result.submissions
                   << ", \"checksum\": " << json_quote(hex_hash(result.output.hash))
                   << ", \"repeat_checksum\": " << json_quote(hex_hash(result.repeat_hash)) << "}";
        }
        const CompareStats & compare = cases[case_index].normal_vs_swiglu;
        output << "\n    }, \"normal_vs_swiglu\": {\"max_abs\": " << compare.max_abs
               << ", \"max_rel\": " << compare.max_rel << "}}";
    }
    output << "\n  ]\n}\n";
    output.flush();
    if (!output) { std::fprintf(stderr, "failed writing JSON receipt\n"); std::exit(8); }
}


} // namespace

int main(int argc, char ** argv) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }
    if (options.model.empty()) options.model = default_model_path();

    ggml_backend_t backend = nullptr;
    if (options.backend == "gpu") {
#if BREEZE_FFN_VULKAN
        backend = ggml_backend_vk_init(0);
#else
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
#endif
    } else {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) {
        std::fprintf(stderr, "no %s backend is available\n", options.backend.c_str());
        return 3;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    std::printf("backend=%s device=%s model=%s layers=%d warmup=%d iterations=%d\n",
                ggml_backend_name(backend), device ? ggml_backend_dev_name(device) : "unknown",
                options.model.c_str(), options.layers, options.warmup, options.iterations);
    std::printf("weights=real_model_Q4_K input=synthetic_normalized_f32 scale=0.25 "
                "scope=graph_compute+synchronize; split=5 submissions/FFN; batched=independent layer roots\n");

    ModelWeights model;
    if (!load_weights(options.model, backend, model)) {
        ggml_backend_free(backend);
        return 4;
    }

    const size_t graph_context_bytes = ggml_tensor_overhead() * 512
        + ggml_graph_overhead_custom(kLayers * 8, false) * 256 + 8 * 1024 * 1024;
    ggml_context * graph_ctx = make_context(graph_context_bytes);
    std::array<FfnCase, 2> cases = {
        build_case(graph_ctx, model, 1, options.layers),
        build_case(graph_ctx, model, 2, options.layers),
    };
    ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
    if (!graph_buffer) {
        std::fprintf(stderr, "graph allocation failed\n");
        ggml_free(graph_ctx);
        ggml_backend_free(backend);
        return 5;
    }
    for (FfnCase & ffn : cases) {
        std::vector<float> input(static_cast<size_t>(kHidden) * ffn.n);
        fill_normalized(input, 0x243f6a1du + static_cast<uint32_t>(ffn.n));
        ggml_backend_tensor_set(ffn.input, input.data(), 0, input.size() * sizeof(float));
    }

    std::array<CaseResults, 2> results{};
    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        FfnCase & ffn = cases[case_index];
        const TimedResult split = run_variant(backend, ffn, Variant::Split,
                                              options.layers, options.warmup, options.iterations);
        const TimedResult normal = run_variant(backend, ffn, Variant::PerFfn,
                                               options.layers, options.warmup, options.iterations);
        const TimedResult batched = run_variant(backend, ffn, Variant::Batched,
                                                options.layers, options.warmup, options.iterations);
        const TimedResult fused = run_variant(backend, ffn, Variant::PerFfnSwiglu,
                                              options.layers, options.warmup, options.iterations);
        const TimedResult fused_batch = run_variant(backend, ffn, Variant::BatchedSwiglu,
                                                    options.layers, options.warmup, options.iterations);
        print_result(ffn, Variant::Split, split, options.layers);
        print_result(ffn, Variant::PerFfn, normal, options.layers);
        print_result(ffn, Variant::Batched, batched, options.layers);
        print_result(ffn, Variant::PerFfnSwiglu, fused, options.layers);
        print_result(ffn, Variant::BatchedSwiglu, fused_batch, options.layers);
        results[case_index].values = {split, normal, batched, fused, fused_batch};
        results[case_index].normal_vs_swiglu = compare_results(normal, fused);
        print_tolerance(ffn, normal, fused, results[case_index].normal_vs_swiglu);
        if (split.output.hash != normal.output.hash || normal.output.hash != batched.output.hash ||
            fused.output.hash != fused_batch.output.hash ||
            results[case_index].normal_vs_swiglu.max_abs > 1e-5 ||
            results[case_index].normal_vs_swiglu.max_rel > 1e-5) {
            std::fprintf(stderr, "FFN variants failed output-equivalence gate\n");
            return 9;
        }
    }

    write_json(options, device ? ggml_backend_dev_name(device) : "unknown", results);

    ggml_backend_buffer_free(graph_buffer);
    ggml_free(graph_ctx);
    ggml_backend_free(backend);
    return 0;
}
