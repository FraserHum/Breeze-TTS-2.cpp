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
    // frames per streamed chunk, ramping from first to max. set both the same for a fixed size
    int chunk_first = 4;
    int chunk_max = 25;
    // long text is generated in pieces of about this many characters, 0 keeps it in one pass.
    // the model loses track of the text somewhere past a minute of audio, so pieces stay under that
    int split_chars = 600;
};

// break text on sentence boundaries into pieces worth roughly budget characters, cjk counted heavier.
// first_budget sizes the opening piece only, 0 to treat it like the rest
std::vector<std::string> split_text(const std::string & text, int budget, int first_budget = 0);

// rough spoken length in seconds, only good enough to drive a progress bar
double estimate_seconds(const std::string & text);

// how faithfully the acoustic codes are resynthesised. codec language models are known to go
// unstable on random sampling, so conversion leans much harder on the likeliest codes than tts does
struct ConvertOptions {
    std::string src_text;   // transcript of the source, empty generates filler
    float temperature = 0.9f;
    int top_k = 50;
    float cfg_scale = 1.0f;
    // acoustic codebooks taken straight from the source instead of being regenerated. pitch lives in
    // the low ones, so this trades voice identity back for the source's melody and intonation
    int keep_acoustic = 0;
    bool feed_source = false;
    int seed = 42;
};

// respeak already encoded audio in the reference voice. words and frame timing come from the source,
// pitch and timbre from the reference, so intonation is not carried over.
std::vector<float> convert_voice(BreezeModel & m, MimiCodec & codec, const std::vector<int> & src_codes,
                                 int src_T, const std::vector<float> & ref_audio,
                                 const std::string & ref_text, const ConvertOptions & opt = {});

// called with each decoded audio chunk; return false to stop generation early
using AudioCallback = std::function<bool(const float * samples, int n)>;

// per stage wall clock milliseconds, optionally filled by generate
struct GenTimings {
    double encode_ref = 0, prompt = 0, prefill = 0, backbone = 0, depth = 0, vocoder = 0;
    double first_audio = 0, first_vocoder = 0;
    int frames = 0, flushes = 0, first_frames = 0;
};

void generate(BreezeModel & m, MimiCodec & codec, const GenRequest & req, const AudioCallback & cb,
              GenTimings * timings = nullptr);

}
