#include "breeze/gguf_loader.h"

#include <cstdio>
#include <stdexcept>

namespace breeze {

#ifdef _WIN32
#define breeze_fseek _fseeki64
#else
#define breeze_fseek fseeko
#endif

bool GGUFModel::load(const std::string & path, Backend & be) {
    gguf_init_params gp{ /*no_alloc=*/true, /*ctx=*/&meta };
    gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) { fprintf(stderr, "gguf_init_from_file failed for %s\n", path.c_str()); return false; }

    buffer = ggml_backend_alloc_ctx_tensors(meta, be.backend);
    if (!buffer) { fprintf(stderr, "ggml_backend_alloc_ctx_tensors failed\n"); return false; }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "fopen failed for %s\n", path.c_str()); return false; }

    const size_t data_off = gguf_get_data_offset(gguf);
    const int64_t n = gguf_get_n_tensors(gguf);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        const size_t off = data_off + gguf_get_tensor_offset(gguf, i);
        const size_t sz = ggml_nbytes(t);
        buf.resize(sz);
        if (breeze_fseek(f, (long long) off, SEEK_SET) != 0) { fclose(f); return false; }
        if (fread(buf.data(), 1, sz, f) != sz) { fclose(f); return false; }
        ggml_backend_tensor_set(t, buf.data(), 0, sz);
        tensors[name] = t;
    }
    fclose(f);
    return true;
}

bool GGUFModel::load_extra(const std::string & path, Backend & be) {
    ggml_context * extra_meta = nullptr;
    gguf_init_params gp{ /*no_alloc=*/true, /*ctx=*/&extra_meta };
    gguf_context * extra_gguf = gguf_init_from_file(path.c_str(), gp);
    if (!extra_gguf) { fprintf(stderr, "gguf_init_from_file failed for extra %s\n", path.c_str()); return false; }

    ggml_backend_buffer_t extra_buffer = ggml_backend_alloc_ctx_tensors(extra_meta, be.backend);
    if (!extra_buffer) {
        fprintf(stderr, "ggml_backend_alloc_ctx_tensors failed for extra %s\n", path.c_str());
        ggml_free(extra_meta);
        gguf_free(extra_gguf);
        return false;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "fopen failed for %s\n", path.c_str());
        ggml_backend_buffer_free(extra_buffer);
        ggml_free(extra_meta);
        gguf_free(extra_gguf);
        return false;
    }

    const size_t data_off = gguf_get_data_offset(extra_gguf);
    const int64_t n = gguf_get_n_tensors(extra_gguf);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(extra_gguf, i);
        ggml_tensor * t = ggml_get_tensor(extra_meta, name);
        const size_t off = data_off + gguf_get_tensor_offset(extra_gguf, i);
        const size_t sz = ggml_nbytes(t);
        buf.resize(sz);
        if (breeze_fseek(f, (long long) off, SEEK_SET) != 0) { fclose(f); return false; }
        if (fread(buf.data(), 1, sz, f) != sz) { fclose(f); return false; }
        ggml_backend_tensor_set(t, buf.data(), 0, sz);
        tensors[name] = t;
    }
    fclose(f);
    extra_ggufs.push_back(extra_gguf);
    extra_metas.push_back(extra_meta);
    extra_buffers.push_back(extra_buffer);
    return true;
}

void GGUFModel::free() {
    for (auto * eb : extra_buffers) ggml_backend_buffer_free(eb);
    for (auto * em : extra_metas) ggml_free(em);
    for (auto * eg : extra_ggufs) gguf_free(eg);
    extra_buffers.clear();
    extra_metas.clear();
    extra_ggufs.clear();

    if (packed_buffer) ggml_backend_buffer_free(packed_buffer);
    if (packed_meta) ggml_free(packed_meta);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (meta) ggml_free(meta);
    if (gguf) gguf_free(gguf);
    packed_buffer = nullptr;
    packed_meta = nullptr;
    buffer = nullptr;
    meta = nullptr;
    gguf = nullptr;
    tensors.clear();
}

bool GGUFModel::pack_weights(const BreezeConfig & cfg, Backend & be) {
    if (has("bb.blk.0.attn_qkv.weight")) {
        return true; // already packed in GGUF
    }

    struct PackPlan {
        std::string out_name;
        std::vector<std::string> in_names;
    };
    std::vector<PackPlan> plans;

    // Backbone layers
    for (int il = 0; il < cfg.bb.n_layer; il++) {
        const std::string p = "bb.blk." + std::to_string(il);
        if (has(p + ".attn_q.weight") && has(p + ".attn_k.weight") && has(p + ".attn_v.weight")) {
            plans.push_back({p + ".attn_qkv.weight", {p + ".attn_q.weight", p + ".attn_k.weight", p + ".attn_v.weight"}});
        }
        if (has(p + ".ffn_gate.weight") && has(p + ".ffn_up.weight")) {
            plans.push_back({p + ".ffn_gate_up.weight", {p + ".ffn_gate.weight", p + ".ffn_up.weight"}});
        }
    }

    // Depth decoder layers
    for (int il = 0; il < cfg.dd.n_layer; il++) {
        const std::string p = "dd.blk." + std::to_string(il);
        if (has(p + ".attn_q.weight") && has(p + ".attn_k.weight") && has(p + ".attn_v.weight")) {
            plans.push_back({p + ".attn_qkv.weight", {p + ".attn_q.weight", p + ".attn_k.weight", p + ".attn_v.weight"}});
        }
        if (has(p + ".ffn_gate.weight") && has(p + ".ffn_up.weight")) {
            plans.push_back({p + ".ffn_gate_up.weight", {p + ".ffn_gate.weight", p + ".ffn_up.weight"}});
        }
    }

    if (plans.empty()) return true;

    const size_t n_tensors = plans.size();
    const size_t meta_size = n_tensors * (ggml_tensor_overhead() + 1024) + 1024 * 1024;
    ggml_init_params params = { meta_size, nullptr, true };
    packed_meta = ggml_init(params);
    if (!packed_meta) return false;

    std::vector<ggml_tensor *> packed_tensors;
    packed_tensors.reserve(plans.size());

    for (const auto & plan : plans) {
        ggml_tensor * first = get(plan.in_names[0]);
        int64_t in_dim = first->ne[0];
        int64_t total_out = 0;
        for (const auto & in_name : plan.in_names) {
            ggml_tensor * t = get(in_name);
            if (t->ne[0] != in_dim || t->type != first->type) {
                ggml_free(packed_meta);
                packed_meta = nullptr;
                return false;
            }
            total_out += t->ne[1];
        }
        ggml_tensor * t_packed = ggml_new_tensor_2d(packed_meta, first->type, in_dim, total_out);
        if (!t_packed) {
            ggml_free(packed_meta);
            packed_meta = nullptr;
            return false;
        }
        ggml_set_name(t_packed, plan.out_name.c_str());
        packed_tensors.push_back(t_packed);
    }

    packed_buffer = ggml_backend_alloc_ctx_tensors(packed_meta, be.backend);
    if (!packed_buffer) {
        ggml_free(packed_meta);
        packed_meta = nullptr;
        return false;
    }

    std::vector<uint8_t> part_buf;
    std::vector<uint8_t> packed_buf;
    for (size_t i = 0; i < plans.size(); i++) {
        const auto & plan = plans[i];
        ggml_tensor * t_packed = packed_tensors[i];
        packed_buf.clear();

        for (const auto & in_name : plan.in_names) {
            ggml_tensor * in_t = get(in_name);
            const size_t sz = ggml_nbytes(in_t);
            part_buf.resize(sz);
            ggml_backend_tensor_get(in_t, part_buf.data(), 0, sz);
            packed_buf.insert(packed_buf.end(), part_buf.begin(), part_buf.end());
        }

        if (packed_buf.size() != ggml_nbytes(t_packed)) {
            ggml_backend_buffer_free(packed_buffer);
            ggml_free(packed_meta);
            packed_buffer = nullptr;
            packed_meta = nullptr;
            return false;
        }

        ggml_backend_tensor_set(t_packed, packed_buf.data(), 0, packed_buf.size());
        tensors[plan.out_name] = t_packed;
    }

    return true;
}

ggml_tensor * GGUFModel::find(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor * GGUFModel::get(const std::string & name) const {
    ggml_tensor * t = find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

bool GGUFModel::has(const std::string & name) const {
    return tensors.count(name) > 0;
}

int GGUFModel::kv_u32(const char * key, int def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT32:  return (int) gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(gguf, id);
        case GGUF_TYPE_INT64:  return (int) gguf_get_val_i64(gguf, id);
        default: return def;
    }
}

float GGUFModel::kv_f32(const char * key, float def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    const gguf_type t = gguf_get_kv_type(gguf, id);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gguf, id);
    if (t == GGUF_TYPE_FLOAT64) return (float) gguf_get_val_f64(gguf, id);
    return def;
}

bool GGUFModel::kv_bool(const char * key, bool def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_BOOL) return gguf_get_val_bool(gguf, id);
    return def;
}

std::string GGUFModel::kv_str(const char * key, const std::string & def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(gguf, id);
}

std::vector<int> GGUFModel::kv_i32_array(const char * key) const {
    std::vector<int> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    const void * data = gguf_get_arr_data(gguf, id);
    const gguf_type at = gguf_get_arr_type(gguf, id);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (at == GGUF_TYPE_INT32) out[i] = ((const int32_t *) data)[i];
        else if (at == GGUF_TYPE_UINT32) out[i] = (int) ((const uint32_t *) data)[i];
    }
    return out;
}

std::vector<std::string> GGUFModel::kv_str_array(const char * key) const {
    std::vector<std::string> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    out.reserve(n);
    for (size_t i = 0; i < n; i++) out.emplace_back(gguf_get_arr_str(gguf, id, i));
    return out;
}

std::vector<uint8_t> GGUFModel::kv_bytes(const char * key) const {
    std::vector<uint8_t> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(gguf, id) != GGUF_TYPE_UINT8) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    const auto * data = (const uint8_t *) gguf_get_arr_data(gguf, id);
    out.assign(data, data + n);
    return out;
}

}
