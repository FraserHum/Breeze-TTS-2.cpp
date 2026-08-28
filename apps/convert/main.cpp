#include "breeze/audio.h"
#include "breeze/codec.h"
#include "breeze/generation.h"
#include "breeze/model.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace breeze;

static void usage() {
    printf("usage: breeze-convert <model.gguf> --source <in.wav> --output <out.wav> [options]\n"
           "\n"
           "  --source <wav>      speech to convert\n"
           "  --output <wav>      where to write the result (default convert.wav)\n"
           "  --ref-audio <wav>   target voice to convert into\n"
           "  --ref-text <text>   exact transcript of the reference\n"
           "  --text <text>       transcript of the source, lines the backbone up with the\n"
           "                      forced codes. leave it out to run textless\n"
           "  --temp <f>          depth sampling temperature (default 0.3, lower is more faithful)\n"
           "  --top-k <n>         depth top-k, 1 is greedy (default 1). opening this up lets the\n"
           "                      source voice bleed back through\n"
           "  --cfg-scale <f>     guidance toward the target voice (default 1.0, off)\n"
           "  --seed <n>          random seed (default 42)\n"
           "  --keep-acoustic <n> acoustic codebooks kept from the source, 0 to 15. pitch lives in\n"
           "                      the low ones, so raising this keeps the source melody and\n"
           "                      intonation but lets its voice back in (default 0)\n"
           "  --feed-source       feed the original frames back instead of the converted ones\n"
           "  --keep <n>          rebuild from the first n codebooks only, no voice change.\n"
           "                      1 is the semantic stage on its own\n"
           "  --cpu               force the cpu backend\n");
}

static const char * arg(int argc, char ** argv, int & i, const char * name) {
    if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(1); }
    return argv[++i];
}

int main(int argc, char ** argv) {
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage();
        return argc < 2 ? 1 : 0;
    }
    std::string model_path = argv[1], source, output = "convert.wav", ref_audio, ref_text;
    ConvertOptions opt;
    opt.temperature = 0.3f;
    opt.top_k = 1;
    int keep = 0;
    bool use_gpu = true;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--source") source = arg(argc, argv, i, "--source");
        else if (a == "--output") output = arg(argc, argv, i, "--output");
        else if (a == "--ref-audio") ref_audio = arg(argc, argv, i, "--ref-audio");
        else if (a == "--ref-text") ref_text = arg(argc, argv, i, "--ref-text");
        else if (a == "--text") opt.src_text = arg(argc, argv, i, "--text");
        else if (a == "--temp") opt.temperature = (float) atof(arg(argc, argv, i, "--temp"));
        else if (a == "--top-k") opt.top_k = atoi(arg(argc, argv, i, "--top-k"));
        else if (a == "--cfg-scale") opt.cfg_scale = (float) atof(arg(argc, argv, i, "--cfg-scale"));
        else if (a == "--keep-acoustic") opt.keep_acoustic = atoi(arg(argc, argv, i, "--keep-acoustic"));
        else if (a == "--seed") opt.seed = atoi(arg(argc, argv, i, "--seed"));
        else if (a == "--feed-source") opt.feed_source = true;
        else if (a == "--keep") keep = atoi(arg(argc, argv, i, "--keep"));
        else if (a == "--cpu") use_gpu = false;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (source.empty()) { fprintf(stderr, "--source is required\n"); return 1; }

    BreezeModel model;
    printf("loading %s ...\n", model_path.c_str());
    if (!model.load(model_path, use_gpu)) { fprintf(stderr, "failed to load model\n"); return 1; }
    printf("backend: %s\n", model.backend.name());

    MimiCodec codec;
    codec.init(model);

    std::vector<float> src;
    if (!read_wav(source, model.cfg.sample_rate, src)) {
        fprintf(stderr, "failed to read %s\n", source.c_str());
        return 1;
    }
    int T = 0;
    std::vector<int> codes = codec.encode(src, T);
    const int nc = model.cfg.num_codebooks;
    printf("source %.2f s, %d frames\n", (float) src.size() / model.cfg.sample_rate, T);

    std::vector<float> audio;
    if (keep > 0) {
        const int n = keep < nc ? keep : nc;
        std::vector<int> sub((size_t) T * n);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < n; c++) sub[(size_t) t * n + c] = codes[(size_t) t * nc + c];
        printf("rebuilding from %d of %d codebooks\n", n, nc);
        audio = codec.decode(sub, T, n);
    } else {
        std::vector<float> ref;
        if (ref_audio.empty() || ref_text.empty()) {
            fprintf(stderr, "conversion needs --ref-audio and --ref-text\n");
            return 1;
        }
        if (!read_wav(ref_audio, model.cfg.sample_rate, ref)) {
            fprintf(stderr, "failed to read %s\n", ref_audio.c_str());
            return 1;
        }
        audio = convert_voice(model, codec, codes, T, ref, ref_text, opt);
    }

    if (!write_wav(output, audio, model.cfg.sample_rate)) {
        fprintf(stderr, "failed to write %s\n", output.c_str());
        return 1;
    }
    printf("wrote %s (%.2f s)\n", output.c_str(), (float) audio.size() / model.cfg.sample_rate);
    model.free();
    return 0;
}
