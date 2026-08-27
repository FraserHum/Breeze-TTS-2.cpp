#pragma once

#include "breeze/model.h"

#include <vector>

namespace breeze {

// encodes one text segment, returns projected hidden states as [hidden_size * n_tokens] (ne0 = hidden_size)
std::vector<float> text_encoder_forward(BreezeModel & m, const std::vector<int> & tokens);

}
