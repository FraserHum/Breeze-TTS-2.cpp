#pragma once

#include "breeze/common.h"
#include "breeze/config.h"
#include "ggml.h"
#include "gguf.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace breeze {

// loads a GGUF file, uploads all tensor data to the backend and exposes typed KV access
struct GGUFModel {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * packed_meta = nullptr;
    ggml_backend_buffer_t packed_buffer = nullptr;
    std::vector<gguf_context *> extra_ggufs;
    std::vector<ggml_context *> extra_metas;
    std::vector<ggml_backend_buffer_t> extra_buffers;
    std::unordered_map<std::string, ggml_tensor *> tensors;

    bool load(const std::string & path, Backend & be);
    bool load_extra(const std::string & path, Backend & be);
    void free();
    bool pack_weights(const struct BreezeConfig & cfg, Backend & be);

    ggml_tensor * get(const std::string & name) const;
    ggml_tensor * find(const std::string & name) const;
    bool has(const std::string & name) const;

    int   kv_u32(const char * key, int def = 0) const;
    float kv_f32(const char * key, float def = 0.0f) const;
    bool  kv_bool(const char * key, bool def = false) const;
    std::string kv_str(const char * key, const std::string & def = "") const;
    std::vector<int> kv_i32_array(const char * key) const;
    std::vector<std::string> kv_str_array(const char * key) const;
    std::vector<uint8_t> kv_bytes(const char * key) const;
};

}
