// Round-trip test for embedded-voice GGUF metadata.
//
// Builds a synthetic GGUF using the breeze.embedded_voice.* schema (the exact keys the exporter
// writes), then loads it through the real GGUFModel::load + kv_bytes + parse_voice_bytes path and
// asserts the Voice round-trips byte-for-byte against the source .breeze files. Also exercises
// the loader's guard rails (bad name, truncated payload, missing KV -> empty).
#include "breeze/common.h"
#include "breeze/gguf_loader.h"
#include "breeze/voice.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

using namespace breeze;

static int g_fails = 0;
#define CHECK(cond, msg) do { if (cond) { printf("  PASS  %s\n", msg); } else { printf("  FAIL  %s\n", msg); g_fails++; } } while (0)

static void read_file(const char * path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    out.resize((size_t) ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(out.data(), 1, out.size(), f) != out.size()) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
}

// embed one voice as raw BRZV bytes under the documented schema keys
static void embed_voice(gguf_context * ctx, const char * name, const std::vector<uint8_t> & raw) {
    std::string key = "breeze.embedded_voice." + std::string(name);
    gguf_set_arr_data(ctx, key.c_str(), GGUF_TYPE_UINT8, raw.data(), raw.size());
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s calliope.breeze steward.breeze [out.gguf]\n", argv[0]);
        return 2;
    }

    std::vector<uint8_t> call_raw, stew_raw;
    read_file(argv[1], call_raw);
    read_file(argv[2], stew_raw);
    const std::string out_path = (argc > 3) ? argv[3] : "/tmp/test-embed.gguf";

    // --- build the synthetic GGUF with the exporter's exact schema ---
    printf("building synthetic GGUF with embedded voices ...\n");
    gguf_context * ctx = gguf_init_empty();
    const char * arch = "breeze";
    gguf_set_val_str(ctx, "general.architecture", arch);
    gguf_set_val_str(ctx, "general.name", "embedded-voice-test");

    const char * names[] = { "calliope", "steward" };
    gguf_set_arr_str(ctx, "breeze.embedded_voice_names", names, 2);
    embed_voice(ctx, "calliope", call_raw);
    embed_voice(ctx, "steward", stew_raw);
    gguf_set_val_str(ctx, "breeze.embedded_voice.default", "steward");

    // a 1D tensor so GGUFModel::load's tensor loop has something to walk (no model weights needed)
    ggml_init_params p{ 1024 * 1024, nullptr, true };
    ggml_context * meta = ggml_init(p);
    ggml_tensor * w = ggml_new_tensor_1d(meta, GGML_TYPE_F32, 8);
    ggml_set_name(w, "probe.weight");
    const float zeros[8] = { 0 };
    gguf_add_tensor(ctx, w);
    gguf_set_tensor_data(ctx, "probe.weight", zeros);

    if (!gguf_write_to_file(ctx, out_path.c_str(), false)) {
        fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    {
        FILE * w = fopen(out_path.c_str(), "rb");
        if (w) { fseek(w, 0, SEEK_END); printf("wrote %s (%lld bytes)\n", out_path.c_str(), (long long) ftell(w)); fclose(w); }
    }

    // --- load through the real GGUFModel path ---
    printf("loading via GGUFModel::load ...\n");
    Backend backend;
    backend.init(false); // CPU, matches the build (Vulkan off)
    GGUFModel model;
    if (!model.load(out_path, backend)) {
        fprintf(stderr, "GGUFModel::load failed\n");
        return 1;
    }

    printf("checking KV ...\n");
    const std::vector<std::string> names_k = model.kv_str_array("breeze.embedded_voice_names");
    CHECK(names_k.size() == 2, "names array has 2 entries");
    CHECK(names_k.size() == 2 && names_k[0] == "calliope" && names_k[1] == "steward", "names order preserved");
    CHECK(model.kv_str("breeze.embedded_voice.default", "<none>") == "steward", "default = steward");

    const std::vector<uint8_t> call_kv = model.kv_bytes("breeze.embedded_voice.calliope");
    const std::vector<uint8_t> stew_kv = model.kv_bytes("breeze.embedded_voice.steward");
    CHECK(call_kv == call_raw, "calliope kv_bytes is byte-identical to source");
    CHECK(stew_kv == stew_raw, "steward kv_bytes is byte-identical to source");
    CHECK(model.kv_bytes("breeze.embedded_voice.calliops").empty(), "missing key returns empty");

    // parse into Voices and check every field against a canonical load of the same bytes
    Voice calliope, steward;
    bool call_ok = parse_voice_bytes(call_kv.data(), call_kv.size(), "calliope", calliope);
    bool stew_ok = parse_voice_bytes(stew_kv.data(), stew_kv.size(), "steward", steward);
    CHECK(call_ok, "parse_voice_bytes(calliope) succeeds");
    CHECK(stew_ok, "parse_voice_bytes(steward) succeeds");
    // field-by-field against a fresh parse of the original raw (the ground truth)
    Voice gt_call, gt_stew;
    parse_voice_bytes(call_raw.data(), call_raw.size(), "calliope", gt_call);
    parse_voice_bytes(stew_raw.data(), stew_raw.size(), "steward", gt_stew);
    if (call_ok) {
        CHECK(calliope.frames == gt_call.frames, "calliope.frames matches");
        CHECK(calliope.n_codebooks == gt_call.n_codebooks, "calliope.n_codebooks matches");
        CHECK(calliope.sample_rate == gt_call.sample_rate, "calliope.sample_rate matches");
        CHECK(calliope.text == gt_call.text, "calliope.text matches");
        CHECK(calliope.codes == gt_call.codes, "calliope.codes match (vector<int>)");
        CHECK(calliope.name == "calliope", "calliope.name set from key");
    }
    if (stew_ok) {
        CHECK(steward.frames == gt_stew.frames, "steward.frames matches");
        CHECK(steward.n_codebooks == gt_stew.n_codebooks, "steward.n_codebooks matches");
        CHECK(steward.sample_rate == gt_stew.sample_rate, "steward.sample_rate matches");
        CHECK(steward.text == gt_stew.text, "steward.text matches");
        CHECK(steward.codes == gt_stew.codes, "steward.codes match (vector<int>)");
        CHECK(steward.name == "steward", "steward.name set from key");
    }

    // guard rails
    printf("checking guard rails ...\n");
    CHECK(parse_voice_bytes(nullptr, 0, "x", calliope) == false, "empty buffer rejected");
    uint8_t trunc[10];
    memcpy(trunc, call_raw.data(), 10);
    CHECK(parse_voice_bytes(trunc, 10, "x", calliope) == false, "truncated buffer rejected");
    uint8_t badmagic[4];
    memcpy(badmagic, "XXXX", 4);
    CHECK(parse_voice_bytes(badmagic, 4, "x", calliope) == false, "bad magic rejected");

    // full embedded loader (names + per-name bytes + default)
    printf("checking load_embedded_voices ...\n");
    std::string def;
    const std::vector<Voice> all = load_embedded_voices(model, &def);
    CHECK(all.size() == 2, "load_embedded_voices returns 2");
    CHECK(def == "steward", "load_embedded_voices default = steward");
    bool saw_call = false, saw_stew = false;
    for (const Voice & v : all) {
        if (v.name == "calliope") saw_call = true;
        if (v.name == "steward") saw_stew = true;
    }
    CHECK(saw_call && saw_stew, "both embedded voices surfaced");

    model.free();
    backend.free();
    gguf_free(ctx);
    ggml_free(meta);

    printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "ALL PASSED", g_fails);
    return g_fails ? 1 : 0;
}
