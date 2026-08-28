#include "server.h"

#include "breeze/audio.h"
#include "breeze/generation.h"
#include "breeze/model.h"

#include "httplib.h"
#include "breeze_webui_assets.h"

#ifdef _WIN32
#include <windows.h> // after httplib, it pulls in winsock2 first
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace breeze {

static std::string field(const httplib::Request & req, const char * name, const std::string & def) {
    if (req.has_file(name)) return req.get_file_value(name).content;
    if (req.has_param(name)) return req.get_param_value(name);
    return def;
}

struct CachedVoice {
    std::vector<int> codes;
    int frames = 0;
    std::string text;
    double encode_ms = 0;
};

// keyed off the clip and its transcript, so sending the same voice twice lands on the same entry
static std::string voice_key(const std::string & wav, const std::string & text) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : wav) { h ^= c; h *= 1099511628211ull; }
    for (unsigned char c : text) { h ^= c; h *= 1099511628211ull; }
    char buf[32];
    snprintf(buf, sizeof buf, "v_%016llx", (unsigned long long) h);
    return buf;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof b, "\\u%04x", (unsigned) (unsigned char) c);
                    o += b;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

static std::string voice_json(const std::string & id, const CachedVoice & v, int sr, int spf) {
    char buf[256];
    snprintf(buf, sizeof buf, "{\"id\":\"%s\",\"frames\":%d,\"seconds\":%.2f,\"encode_ms\":%.0f,\"ref_text\":\"",
             id.c_str(), v.frames, (double) v.frames * spf / sr, v.encode_ms);
    return std::string(buf) + json_escape(v.text) + "\"}";
}

static std::string mmss(double s) {
    if (!(s >= 0) || s > 86399) s = 0;
    char buf[16];
    snprintf(buf, sizeof buf, "%02d:%02d", (int) s / 60, (int) s % 60);
    return buf;
}

static std::string bar(double frac, int width) {
    static const char * part[] = { " ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589" };
    if (!(frac > 0)) frac = 0;
    if (frac > 1) frac = 1;
    const double filled = frac * width;
    const int full = (int) filled;
    std::string s;
    for (int i = 0; i < full; i++) s += "\u2588";
    if (full < width) {
        s += part[(int) ((filled - full) * 8)];
        for (int i = full + 1; i < width; i++) s += " ";
    }
    return s;
}

int run_server(const ServerOptions & opts) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // otherwise the bar glyphs and any chinese text come out as mojibake
#endif
    BreezeModel model;
    printf("loading %s ...\n", opts.model.c_str());
    if (!model.load(opts.model, opts.use_gpu)) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    printf("backend: %s, sample rate: %d\n", model.backend.name(), model.cfg.sample_rate);
    MimiCodec codec;
    codec.init(model);

    httplib::Server svr;
    auto mutex = std::make_shared<std::mutex>();
    const int sr = model.cfg.sample_rate;
    const int spf = model.cfg.samples_per_frame;

    std::map<std::string, CachedVoice> voices;
    std::vector<std::string> voice_order;
    std::mutex voice_mutex;
    const size_t max_voices = 64;

    // fills in a cached reference, returns false only when the id is unknown
    auto take_voice = [&](const std::string & id, std::vector<int> & codes, int & frames,
                          std::string & text) {
        std::lock_guard<std::mutex> vg(voice_mutex);
        auto it = voices.find(id);
        if (it == voices.end()) return false;
        codes = it->second.codes;
        frames = it->second.frames;
        if (text.empty()) text = it->second.text;
        return true;
    };

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\",\"sample_rate\":" + std::to_string(sr) + "}", "application/json");
    });

    svr.Post("/v1/voices", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string text = field(req, "ref_text", "");
        if (!req.has_file("ref_audio") || text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_audio and ref_text are required\"}", "application/json");
            return;
        }
        const auto & f = req.get_file_value("ref_audio");
        const std::string id = voice_key(f.content, text);
        {
            std::lock_guard<std::mutex> vg(voice_mutex);
            auto it = voices.find(id);
            if (it != voices.end()) {
                res.set_content(voice_json(id, it->second, sr, spf), "application/json");
                return;
            }
        }
        std::vector<float> pcm;
        read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, pcm);
        if (pcm.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"could not read ref_audio\"}", "application/json");
            return;
        }
        // encoding runs on the same device generation does, so it waits its turn
        std::unique_lock<std::mutex> lock(*mutex, std::try_to_lock);
        if (!lock) {
            res.status = 409;
            res.set_content("{\"error\":\"busy\"}", "application/json");
            return;
        }
        CachedVoice v;
        v.text = text;
        const auto t0 = std::chrono::steady_clock::now();
        v.codes = codec.encode(pcm, v.frames);
        v.encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> vg(voice_mutex);
            if (voice_order.size() >= max_voices) {
                voices.erase(voice_order.front());
                voice_order.erase(voice_order.begin());
            }
            voices[id] = v;
            voice_order.push_back(id);
        }
        printf("voice %s cached, %d frames in %.0f ms\n", id.c_str(), v.frames, v.encode_ms);
        fflush(stdout);
        res.set_content(voice_json(id, v, sr, spf), "application/json");
    });

    svr.Get("/v1/voices", [&](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> vg(voice_mutex);
        std::string out = "[";
        for (size_t i = 0; i < voice_order.size(); i++) {
            if (i) out += ",";
            out += voice_json(voice_order[i], voices[voice_order[i]], sr, spf);
        }
        res.set_content(out + "]", "application/json");
    });

    svr.Delete(R"(/v1/voices/(.+))", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> vg(voice_mutex);
        if (!voices.erase(id)) {
            res.status = 404;
            res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
            return;
        }
        voice_order.erase(std::remove(voice_order.begin(), voice_order.end(), id), voice_order.end());
        res.set_content("{\"deleted\":\"" + id + "\"}", "application/json");
    });

    svr.Post("/v1/audio/speech", [&, mutex](const httplib::Request & req, httplib::Response & res) {
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(*mutex, std::try_to_lock);
        if (!*lock) {
            res.status = 409;
            res.set_content("{\"error\":\"busy\"}", "application/json");
            return;
        }
        GenRequest g;
        g.text = field(req, "text", "");
        g.instruction = field(req, "instruction", "Speak clearly and naturally.");
        g.ref_text = field(req, "ref_text", "");
        g.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
        g.seed = atoi(field(req, "seed", "42").c_str());
        g.temperature = (float) atof(field(req, "temperature", "0").c_str());
        g.top_k = atoi(field(req, "top_k", "0").c_str());
        g.top_p = (float) atof(field(req, "top_p", "0").c_str());
        g.repetition_penalty = (float) atof(field(req, "repetition_penalty", "0").c_str());
        g.max_new_tokens = atoi(field(req, "max_new_tokens", "0").c_str());
        g.split_chars = atoi(field(req, "split_chars", "600").c_str());
        g.chunk_first = opts.chunk_first;
        g.chunk_max = opts.chunk_max;
        if (req.has_file("ref_audio")) {
            const auto & f = req.get_file_value("ref_audio");
            if (!f.content.empty())
                read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, g.ref_audio);
        }
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty() && !take_voice(vid, g.ref_codes, g.ref_frames, g.ref_text)) {
            res.status = 404;
            res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
            return;
        }
        if (g.text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"text is required\"}", "application/json");
            return;
        }

        res.set_header("X-Sample-Rate", std::to_string(sr));
        res.set_header("X-Sample-Format", "s16le");
        res.set_header("Cache-Control", "no-store");

        const bool sent_ins = req.has_file("instruction") || req.has_param("instruction");
        const char * mode = !g.ref_audio.empty() ? (sent_ins ? "direction" : "clone") : "design";
        printf("gen  %s, %d chars, cfg %.1f, seed %d\n", mode, (int) g.text.size(), g.cfg_scale, g.seed);
        fflush(stdout);

        res.set_chunked_content_provider(
            "audio/pcm",
            [&model, &codec, g, lock, sr, verbose = opts.verbose](size_t, httplib::DataSink & sink) {
                GenTimings tm;
                const auto t0 = std::chrono::steady_clock::now();
                const auto elapsed = [&] {
                    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                };
                // the model decides when to stop, so the total is only ever an estimate
                const double est = estimate_seconds(g.text);
                size_t total = 0;
                try {
                    generate(model, codec, g, [&](const float * s, int n) {
                        std::vector<uint8_t> pcm = to_pcm16(s, n);
                        if (!sink.write((const char *) pcm.data(), pcm.size())) return false;
                        total += (size_t) n;
                        const double secs = (double) total / sr, wall = elapsed();
                        const double rate = wall > 0 ? secs / wall : 0;
                        const double frac = est > 0 ? secs / est : 0;
                        const double eta = rate > 0 ? (est > secs ? (est - secs) / rate : 0) : 0;
                        printf("\r%3.0f%%|%s| %.1f/%.1fs [%s<%s, %.1f fps, %.2fx]  ",
                               (frac < 1 ? frac : 1) * 100, bar(frac, 24).c_str(), secs, est,
                               mmss(wall).c_str(), mmss(eta).c_str(), secs * 12.5 / wall, rate);
                        fflush(stdout);
                        return true;
                    }, &tm);
                } catch (const std::exception & e) {
                    fprintf(stderr, "\ngeneration error: %s\n", e.what());
                }
                const double secs = (double) total / sr, wall = elapsed();
                printf("\r%3.0f%%|%s| %.1f/%.1fs [%s, %.1f fps, %.2fx]        \n",
                       100.0, bar(1, 24).c_str(), secs, secs, mmss(wall).c_str(),
                       wall > 0 ? secs * 12.5 / wall : 0.0, wall > 0 ? secs / wall : 0.0);
                printf("     %d frames in %d flushes, first audio %.0f ms\n",
                       tm.frames, tm.flushes, tm.first_audio);
                if (verbose && tm.frames > 0) {
                    printf("     ref %.0f  prompt %.0f  prefill %.0f ms | per frame: backbone %.2f"
                           "  depth %.2f  vocoder %.2f ms\n",
                           tm.encode_ref, tm.prompt, tm.prefill, tm.backbone / tm.frames,
                           tm.depth / tm.frames, tm.vocoder / tm.frames);
                }
                fflush(stdout);
                sink.done();
                return true;
            });
    });

    svr.Post("/v1/audio/convert", [&, mutex](const httplib::Request & req, httplib::Response & res) {
        std::unique_lock<std::mutex> lock(*mutex, std::try_to_lock);
        if (!lock) {
            res.status = 409;
            res.set_content("{\"error\":\"busy\"}", "application/json");
            return;
        }
        std::vector<float> src, ref;
        const auto load = [&](const char * name, std::vector<float> & out) {
            if (!req.has_file(name)) return false;
            const auto & f = req.get_file_value(name);
            return !f.content.empty() &&
                   read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, out);
        };
        if (!load("source", src)) {
            res.status = 400;
            res.set_content("{\"error\":\"a source wav file is required\"}", "application/json");
            return;
        }
        std::string ref_text = field(req, "ref_text", "");
        std::vector<int> vcodes;
        int vframes = 0;
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty()) {
            if (!take_voice(vid, vcodes, vframes, ref_text)) {
                res.status = 404;
                res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
                return;
            }
        } else if (!load("ref_audio", ref)) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_audio or voice_id is required\"}", "application/json");
            return;
        }
        if (ref_text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_text is required\"}", "application/json");
            return;
        }

        printf("conv %.2f s source, %s\n", (double) src.size() / sr,
               vframes > 0 ? "cached reference" : "uploaded reference");
        fflush(stdout);
        try {
            int T = 0;
            std::vector<int> codes = codec.encode(src, T);
            ConvertOptions copt;
            copt.src_text = field(req, "text", "");
            copt.temperature = (float) atof(field(req, "temperature", "0.3").c_str());
            copt.top_k = atoi(field(req, "top_k", "1").c_str());
            copt.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
            copt.keep_acoustic = atoi(field(req, "keep_acoustic", "0").c_str());
            copt.seed = atoi(field(req, "seed", "42").c_str());
            copt.ref_codes = vcodes;
            copt.ref_frames = vframes;
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> audio = convert_voice(model, codec, codes, T, ref, ref_text, copt);
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double secs = (double) audio.size() / sr;
            printf("     %.2f s in %.2f s, %.2fx rt\n", secs, wall, wall > 0 ? secs / wall : 0.0);
            fflush(stdout);
            std::vector<uint8_t> pcm = to_pcm16(audio.data(), (int) audio.size());
            res.set_header("X-Sample-Rate", std::to_string(sr));
            res.set_header("X-Sample-Format", "s16le");
            res.set_content((const char *) pcm.data(), pcm.size(), "audio/pcm");
        } catch (const std::exception & e) {
            fprintf(stderr, "conversion error: %s\n", e.what());
            res.status = 500;
            res.set_content("{\"error\":\"conversion failed\"}", "application/json");
        }
    });

    if (opts.webui) {
        svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::index_html, "text/html");
        });
        svr.Get("/style.css", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::style_css, "text/css");
        });
        svr.Get("/app.js", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::app_js, "application/javascript");
        });
        printf("web ui: http://%s:%d/\n", opts.host.c_str(), opts.port);
    }

    printf("listening on http://%s:%d\n", opts.host.c_str(), opts.port);
    if (!svr.listen(opts.host, opts.port)) {
        fprintf(stderr, "failed to bind %s:%d\n", opts.host.c_str(), opts.port);
        return 1;
    }
    model.free();
    return 0;
}

}
