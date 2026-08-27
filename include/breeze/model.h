#pragma once

#include "breeze/config.h"
#include "breeze/common.h"
#include "breeze/gguf_loader.h"
#include "breeze/tokenizer.h"

#include <memory>
#include <string>

namespace breeze {

struct BreezeModel {
    Backend backend;
    GGUFModel gg;
    BreezeConfig cfg;
    Tokenizer tok;

    bool load(const std::string & path, bool prefer_gpu);
    void free();

    ggml_tensor * w(const std::string & name) const { return gg.get(name); }
    ggml_tensor * wopt(const std::string & name) const { return gg.find(name); }
};

}
