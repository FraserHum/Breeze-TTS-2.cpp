#pragma once

#include "breeze/backbone.h"
#include "breeze/codec.h"
#include "breeze/depth_decoder.h"
#include "breeze/model.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace breeze {

struct GenRequest {
    std::string text;
    std::string instruction = "Speak clearly and naturally.";
    std::string ref_text;
    std::vector<float> ref_audio; // mono 24 kHz, empty for voice design
    // already encoded reference. takes priority over ref_audio so a repeated voice skips the codec,
    // which is most of the wait before the first audio comes out
    std::vector<int> ref_codes;
    int ref_frames = 0;
    // saved voice name. when its prefix is held in the store, begin() restores that snapshot
    // instead of prefilling the reference; empty for no-reference and inline-reference requests
    std::string voice;
    float cfg_scale = 1.0f;
    int seed = 42;
    // sampling. zero on any of these keeps whatever the gguf was built with
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 0.0f;
    float repetition_penalty = 0.0f;
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
    float temperature = 0.3f;
    int top_k = 1;
    float cfg_scale = 1.0f;
    // acoustic codebooks taken straight from the source instead of being regenerated. pitch lives in
    // the low ones, so this trades voice identity back for the source's melody and intonation
    int keep_acoustic = 0;
    bool feed_source = false;
    int seed = 42;
    // already encoded reference, used instead of the ref_audio argument when set
    std::vector<int> ref_codes;
    int ref_frames = 0;
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
    // backbone/depth wall time accumulated at the moment of the first flush. with encode_ref,
    // prompt, prefill and first_vocoder they add up to time-to-first-audio (BREEZE_TTA_BREAKDOWN)
    double bb_first = 0, depth_first = 0;
    int frames = 0, flushes = 0, first_frames = 0;
};

void generate(BreezeModel & m, MimiCodec & codec, const GenRequest & req, const AudioCallback & cb,
              GenTimings * timings = nullptr);

// prefill the reference with the exact op sequence begin() runs per generate and hold a snapshot of
// its k/v under the voice name, so requests for that voice restore the snapshot and pay only their
// own tail prefill. returns the snapshot bytes, or 0 when the voice could not be built (it then
// keeps the per-generate prefill). the cli calls this once after loading a saved voice and the
// server for every voice at startup
size_t build_voice_prefix(BreezeModel & m, const std::string & name,
                          const std::vector<int> & ref_codes, const std::string & ref_text,
                          int ref_frames);

// generation spread over several calls, for text that arrives a bit at a time. it holds the
// reference so the voice does not drift between pieces, and the instruction can change as it goes
class GenSession {
public:
    void begin(BreezeModel & m, MimiCodec & codec, const GenRequest & req, GenTimings * tm = nullptr);

    // speaks one piece. false means the callback asked to stop
    bool speak(const std::string & text, const AudioCallback & cb, GenTimings * tm = nullptr);

    void set_instruction(const std::string & s) { m_req.instruction = s; }

    // with no clip to clone the first piece becomes the reference, so it wants to stay short
    bool needs_anchor() const { return m_codes.empty(); }

private:
    BreezeModel * m_model = nullptr;
    MimiCodec * m_codec = nullptr;
    GenRequest m_req;
    std::vector<int> m_codes;
    std::string m_text;
    int m_frames = 0;
    uint32_t m_piece = 0;
    std::chrono::steady_clock::time_point m_start;
    // reference prefix k/v built once in begin(); each part restores it instead of re-encoding and
    // re-prefilling the reference. empty when there is no reference to clone
    std::vector<std::vector<float>> m_prefix;
    int m_prefix_len = 0;
};

}
