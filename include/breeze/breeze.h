#ifndef BREEZE_H
#define BREEZE_H

#include <stddef.h>

#if defined(_WIN32)
#ifdef BREEZE_BUILD_SHARED
#define BREEZE_API __declspec(dllexport)
#else
#define BREEZE_API
#endif
#else
#define BREEZE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct breeze_context breeze_context;

// return 0 to continue, non-zero to stop generation
typedef int (*breeze_audio_cb)(const float * samples, int n_samples, void * user);

typedef struct {
    const char * text;
    const char * instruction;   // null for a neutral default
    const char * ref_text;      // null when not cloning
    const float * ref_audio;    // mono 24 kHz samples, null for voice design
    int ref_audio_len;
    float cfg_scale;
    int seed;
    int max_new_tokens;         // 0 uses the model default
    int split_chars;            // 0 uses the default, negative keeps long text in a single pass
} breeze_request;

BREEZE_API breeze_context * breeze_init(const char * gguf_path, int use_gpu);
BREEZE_API void breeze_free(breeze_context * ctx);
BREEZE_API int breeze_sample_rate(breeze_context * ctx);

// streams audio chunks to the callback; returns 0 on success
BREEZE_API int breeze_generate(breeze_context * ctx, const breeze_request * req,
                               breeze_audio_cb cb, void * user);

// convenience: generate and write a 16-bit PCM WAV; returns 0 on success
BREEZE_API int breeze_generate_wav(breeze_context * ctx, const breeze_request * req,
                                   const char * out_path);

BREEZE_API const char * breeze_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
