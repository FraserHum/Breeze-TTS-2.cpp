#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace breeze {

// reads a WAV file, returns mono float samples resampled to target_sr
bool read_wav(const std::string & path, int target_sr, std::vector<float> & out);

// parses a WAV from an in-memory buffer, returns mono float resampled to target_sr
bool read_wav_buffer(const uint8_t * data, size_t size, int target_sr, std::vector<float> & out);

// writes float samples as 16-bit PCM mono WAV
bool write_wav(const std::string & path, const std::vector<float> & samples, int sr);

// float samples in [-1, 1] to signed 16-bit little-endian bytes
std::vector<uint8_t> to_pcm16(const float * samples, int n);

std::vector<float> resample_linear(const std::vector<float> & in, int in_sr, int out_sr);

}
