#include "breeze/audio.h"
#include "breeze/codec.h"
#include "breeze/model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using namespace breeze;

struct Stats {
    bool finite = true;
    bool equal_length = true;
    double snr = 0.0;
    double corr = 0.0;
    double max_diff = 0.0;
};

static Stats compare(const std::vector<float> & ref, size_t ref_off,
                     const std::vector<float> & got, size_t got_off, size_t n) {
    Stats s;
    if (ref.size() < ref_off + n || got.size() < got_off + n) {
        s.equal_length = false;
        return s;
    }
    double ma = 0.0, mb = 0.0;
    for (size_t i = 0; i < n; i++) {
        const float a = ref[ref_off + i], b = got[got_off + i];
        if (!std::isfinite(a) || !std::isfinite(b)) s.finite = false;
        ma += a;
        mb += b;
        const double d = std::abs((double) a - b);
        if (d > s.max_diff) s.max_diff = d;
    }
    if (!s.finite) return s;
    ma /= n;
    mb /= n;
    double aa = 0.0, bb = 0.0, ab = 0.0, signal = 0.0, noise = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double a = ref[ref_off + i], b = got[got_off + i];
        const double da = a - ma, db = b - mb;
        aa += da * da;
        bb += db * db;
        ab += da * db;
        signal += a * a;
        const double d = a - b;
        noise += d * d;
    }
    s.corr = aa > 0.0 && bb > 0.0 ? ab / std::sqrt(aa * bb) : (s.max_diff == 0.0 ? 1.0 : 0.0);
    s.snr = noise == 0.0 ? std::numeric_limits<double>::infinity()
                         : 10.0 * std::log10(signal / noise);
    return s;
}

int main(int argc, char ** argv) {
    bool use_gpu = true;
    if (argc == 4 && std::string(argv[3]) == "--cpu") use_gpu = false;
    else if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <speech.wav> [--cpu]\n", argv[0]);
        return 2;
    }
    if (setenv("BREEZE_VOC_TRIM", "1", 1) != 0) {
        std::fprintf(stderr, "failed to enable BREEZE_VOC_TRIM\n");
        return 2;
    }

    BreezeModel model;
    if (!model.load(argv[1], use_gpu)) {
        std::fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    std::vector<float> speech;
    if (!read_wav(argv[2], model.cfg.sample_rate, speech)) {
        std::fprintf(stderr, "failed to read %s\n", argv[2]);
        model.free();
        return 1;
    }

    MimiCodec codec;
    codec.init(model);
    int encoded_frames = 0;
    const std::vector<int> codes = codec.encode(speech, encoded_frames);
    const int nc = model.cfg.num_codebooks;
    const int spf = model.cfg.samples_per_frame;
    constexpr int needed_frames = 128;
    if (nc <= 0 || spf <= 0 || encoded_frames < needed_frames ||
        codes.size() < (size_t) needed_frames * nc) {
        std::fprintf(stderr, "fixture encoded to %d frames; need at least %d\n", encoded_frames, needed_frames);
        model.free();
        return 1;
    }

    const int starts[] = { 0, 16, 88 };
    const int counts[] = { 1, 4, 40 };
    bool ok = true;
    for (int start : starts) {
        for (int count : counts) {
            if (start < 0 || count <= 0 || start + count > needed_frames) {
                std::fprintf(stderr, "invalid test range start=%d count=%d\n", start, count);
                ok = false;
                continue;
            }
            const int T = start + count;
            const std::vector<int> window(codes.begin(), codes.begin() + (size_t) T * nc);
            const std::vector<float> ref = codec.decode(window, T, nc);
            const std::vector<float> got = codec.decode(window, T, nc, start);
            const size_t want = (size_t) count * spf;
            const size_t expected = (size_t) (std::min(start, 16) + count) * spf;
            const bool lengths = ref.size() == (size_t) T * spf && got.size() == expected;
            Stats s;
            if (lengths) s = compare(ref, (size_t) start * spf, got, got.size() - want, want);
            const bool pass = lengths && s.equal_length && s.finite && s.snr >= 30.0 &&
                              s.corr >= 0.999 && s.max_diff < 0.03;
            std::printf("start=%d count=%d ref=%zu got=%zu snr_db=%.3f corr=%.9f max_diff=%.6g %s\n",
                        start, count, ref.size(), got.size(), s.snr, s.corr, s.max_diff,
                        pass ? "PASS" : "FAIL");
            ok = ok && pass;
        }
    }
    model.free();
    return ok ? 0 : 1;
}
