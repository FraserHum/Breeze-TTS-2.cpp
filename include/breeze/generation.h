#pragma once

#include "breeze/backbone.h"
#include "breeze/codec.h"
#include "breeze/depth_decoder.h"
#include "breeze/model.h"

#include <functional>
#include <string>
#include <vector>

namespace breeze {

struct GenRequest {
    std::string text;
    std::string instruction = "Speak clearly and naturally.";
    std::string ref_text;
    std::vector<float> ref_audio; // mono 24 kHz, empty for voice design
    float cfg_scale = 1.0f;
    int seed = 42;
    int max_new_tokens = 0; // 0 uses the model default
};

// called with each decoded audio chunk; return false to stop generation early
using AudioCallback = std::function<bool(const float * samples, int n)>;

void generate(BreezeModel & m, MimiCodec & codec, const GenRequest & req, const AudioCallback & cb);

}
