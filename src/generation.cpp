#include "breeze/generation.h"
#include "breeze/sampling.h"
#include "breeze/text_encoder.h"

#include <algorithm>
#include <chrono>
#include <random>

namespace breeze {

struct Seg {
    bool is_text;
    std::vector<int> tokens;
    std::vector<int> codes; // frame-major, for audio segments
    int n_frames = 0;
    bool eos = false;
};

static Seg text_seg(BreezeModel & m, const std::string & s) {
    Seg seg;
    seg.is_text = true;
    seg.tokens = m.tok.encode(s, true);
    return seg;
}

static std::vector<Seg> build_segments(BreezeModel & m, const GenRequest & r, bool has_ref,
                                       const std::vector<int> & ref_codes, int ref_T, bool cond) {
    std::vector<Seg> segs;
    const std::string spk = "[S0]";
    if (has_ref) {
        segs.push_back(text_seg(m, spk + r.ref_text));
        Seg a;
        a.is_text = false;
        a.codes = ref_codes;
        a.n_frames = ref_T;
        a.eos = true;
        segs.push_back(a);
    }
    std::string tail = cond ? spk + "<ins_bos>" + r.instruction + "<ins_eos>" + r.text : spk + r.text;
    segs.push_back(text_seg(m, tail));
    return segs;
}

static std::vector<float> assemble(BreezeModel & m, const std::vector<Seg> & segs, int & total) {
    const int H = m.cfg.hidden_size;
    std::vector<float> out;
    total = 0;
    for (const Seg & s : segs) {
        if (s.is_text) {
            std::vector<float> e = text_encoder_forward(m, s.tokens);
            out.insert(out.end(), e.begin(), e.end());
            total += (int) s.tokens.size();
        } else {
            std::vector<float> e = audio_embed_forward(m, s.codes, s.n_frames);
            out.insert(out.end(), e.begin(), e.end());
            total += s.n_frames;
            std::vector<int> eos_frame(m.cfg.num_codebooks, m.cfg.codebook_eos_token_id);
            std::vector<float> ee = audio_embed_forward(m, eos_frame, 1);
            out.insert(out.end(), ee.begin(), ee.end());
            total += 1;
        }
    }
    (void) H;
    return out;
}

static std::vector<float> combine_logits(const std::vector<float> & cond, const std::vector<float> & unc,
                                         bool use_cfg, float scale) {
    if (!use_cfg) return cond;
    std::vector<float> out(cond.size());
    for (size_t i = 0; i < out.size(); i++) out[i] = unc[i] + scale * (cond[i] - unc[i]);
    return out;
}

void generate(BreezeModel & m, MimiCodec & codec, const GenRequest & req, const AudioCallback & cb,
              GenTimings * timings) {
    GenTimings sink;
    GenTimings & tm = timings ? *timings : sink;
    const auto clock_now = [] { return std::chrono::steady_clock::now(); };
    const auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    const auto t_start = clock_now();

    std::mt19937 rng((uint32_t) req.seed);
    const int nc = m.cfg.num_codebooks;
    const int spf = m.cfg.samples_per_frame;
    const bool has_ref = !req.ref_audio.empty() && !req.ref_text.empty();
    const bool use_cfg = req.cfg_scale != 1.0f;

    std::vector<int> ref_codes;
    int ref_T = 0;
    auto t0 = clock_now();
    if (has_ref) ref_codes = codec.encode(req.ref_audio, ref_T);
    tm.encode_ref = since(t0);

    t0 = clock_now();
    int total_c = 0, total_u = 0;
    std::vector<float> emb_c = assemble(m, build_segments(m, req, has_ref, ref_codes, ref_T, true), total_c);
    std::vector<float> emb_u;
    if (use_cfg) emb_u = assemble(m, build_segments(m, req, has_ref, ref_codes, ref_T, false), total_u);
    tm.prompt = since(t0);

    const int max_new = req.max_new_tokens > 0 ? req.max_new_tokens : m.cfg.max_new_tokens;

    BackboneState st_c, st_u;
    st_c.init(m, total_c + max_new + 8);
    if (use_cfg) st_u.init(m, total_u + max_new + 8);

    t0 = clock_now();
    StepOut o_c = backbone_run(m, st_c, emb_c, total_c);
    StepOut o_u;
    if (use_cfg) o_u = backbone_run(m, st_u, emb_u, total_u);
    tm.prefill = since(t0);

    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);

    SampleParams bp;
    bp.temperature = m.cfg.temperature;
    bp.top_k = m.cfg.top_k;
    bp.top_p = m.cfg.top_p;
    bp.repetition_penalty = m.cfg.repetition_penalty;
    std::vector<int> suppress;
    for (int t = m.cfg.codec_codebook_size; t < m.cfg.audio_vocab_size; t++) suppress.push_back(t);

    std::vector<int> hist;
    std::vector<float> comb = combine_logits(o_c.logits, o_u.logits, use_cfg, req.cfg_scale);
    int cb0 = sample_token(comb, bp, rng, &hist, &suppress);

    std::vector<int> frames;
    int emitted = 0;
    // the first flush is small so audio starts early, then it grows to keep the vocoder efficient
    const int chunk_max = std::max(1, req.chunk_max);
    int chunk = std::min(std::max(1, req.chunk_first), chunk_max);
    // the transformer window plus the slack the vocoder convolutions reach back over
    const int ctx = m.cfg.voc.sliding_window + 16;
    auto flush = [&](bool final_flush) {
        const int have = (int) frames.size() / nc;
        while (have - emitted >= chunk || (final_flush && have > emitted)) {
            const int start = emitted;
            const int count = final_flush ? have - emitted : chunk;
            const int ctx_start = start > ctx ? start - ctx : 0;
            const int sub_T = start + count - ctx_start;
            std::vector<int> sub(frames.begin() + (size_t) ctx_start * nc, frames.begin() + (size_t) (start + count) * nc);
            const auto tv = clock_now();
            std::vector<float> audio = codec.decode(sub, sub_T);
            const double vtime = since(tv);
            tm.vocoder += vtime;
            tm.flushes++;
            const int skip = (start - ctx_start) * spf;
            if (!tm.first_audio) {
                tm.first_vocoder = vtime;
                tm.first_frames = sub_T;
                tm.first_audio = since(t_start);
            }
            if (!cb(audio.data() + skip, count * spf)) return false;
            emitted += count;
            chunk = std::min(chunk + chunk / 3 + 1, chunk_max);
            if (!final_flush && have - emitted < chunk) break;
        }
        return true;
    };

    for (int step = 0; step < max_new; step++) {
        if (cb0 == m.cfg.backbone_eos_token_id) break;
        std::vector<std::vector<float>> hiddens = { o_c.hidden };
        if (use_cfg) hiddens.push_back(o_u.hidden);
        auto td = clock_now();
        std::vector<int> depth_codes = depth.run(m, hiddens, cb0, req.cfg_scale, rng);
        tm.depth += since(td);
        std::vector<int> frame = { cb0 };
        frame.insert(frame.end(), depth_codes.begin(), depth_codes.end());

        bool pad = true;
        for (int c : frame) if (c != m.cfg.codebook_pad_token_id) { pad = false; break; }
        if (!pad) {
            frames.insert(frames.end(), frame.begin(), frame.end());
            tm.frames++;
            if (!flush(false)) return;
        }
        hist.push_back(cb0);

        auto tb = clock_now();
        std::vector<float> ae = audio_embed_forward(m, frame, 1);
        o_c = backbone_run(m, st_c, ae, 1);
        if (use_cfg) o_u = backbone_run(m, st_u, ae, 1);
        tm.backbone += since(tb);
        comb = combine_logits(o_c.logits, o_u.logits, use_cfg, req.cfg_scale);
        cb0 = sample_token(comb, bp, rng, &hist, &suppress);
    }
    flush(true);

    st_c.free();
    if (use_cfg) st_u.free();
    depth.free();
}

}
