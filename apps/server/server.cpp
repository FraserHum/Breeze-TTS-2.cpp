#include "server.h"
#include "voices.h"
#include "ws.h"
#include "ws_api.h"

#include "breeze/audio.h"
#include "breeze/generation.h"
#include "breeze/model.h"
#include "breeze/voice.h"

#include "httplib.h"
#include "breeze_webui_assets.h"

#ifdef _WIN32
#include <windows.h> // after httplib, it pulls in winsock2 first
#endif

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace breeze {

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

    VoiceStore store;
    store.load_dir(opts.voices_dir, model.cfg.num_codebooks);

    // saved voices are fixed for the pod's lifetime, so build each one's reference prefix once here.
    // requests for a known voice then restore the stored snapshot and pay only their own tail; a
    // voice that fails to build logs a warning and keeps the per-generate prefill
    for (const Voice & v : load_voice_dir(opts.voices_dir)) {
        if (v.n_codebooks != model.cfg.num_codebooks) continue;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t bytes = build_voice_prefix(model, v.name, v.codes, v.text, v.frames);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (bytes)
            printf("prefix precompute: voice '%s' %zu bytes in %.0f ms\n", v.name.c_str(), bytes, ms);
        else
            fprintf(stderr, "prefix precompute: voice '%s' failed, keeping the per-generate prefill\n",
                    v.name.c_str());
    }
    fflush(stdout);

    store.add_routes(svr, model, codec, *mutex, opts.voices_dir);

    WsServer ws;
    const int ws_port = opts.ws_port == 0 ? opts.port + 1 : opts.ws_port;
    if (ws_port > 0) {
        const bool up = ws.start(opts.host, ws_port, [&](WsConn & c) {
            ws_connection(c, model, codec, store, *mutex, opts.chunk_first, opts.chunk_max, opts.split_chars);
        });
        if (up) printf("websocket on ws://%s:%d\n", opts.host.c_str(), ws_port);
        else fprintf(stderr, "could not open the websocket port %d\n", ws_port);
    }

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\",\"sample_rate\":" + std::to_string(sr) +
                        ",\"ws_port\":" + std::to_string(ws_port > 0 ? ws_port : 0) + "}",
                        "application/json");
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
        g.split_chars = atoi(field(req, "split_chars", std::to_string(opts.split_chars)).c_str());
        g.chunk_first = opts.chunk_first;
        g.chunk_max = opts.chunk_max;
        if (req.has_file("ref_audio")) {
            const auto & f = req.get_file_value("ref_audio");
            if (!f.content.empty())
                read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, g.ref_audio);
        }
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty() && !store.take(vid, g.ref_codes, g.ref_frames, g.ref_text)) {
            res.status = 404;
            res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
            return;
        }
        if (!vid.empty()) g.voice = vid;
        if (g.text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"text is required\"}", "application/json");
            return;
        }

        const std::string fmt = field(req, "response_format", field(req, "format", "wav"));
        const bool stream = field(req, "stream", "false") == "true";

        res.set_header("X-Sample-Rate", std::to_string(sr));
        res.set_header("X-Sample-Format", "s16le");
        res.set_header("Cache-Control", "no-store");

        const bool sent_ins = !field(req, "instruction", "").empty() || req.has_file("instruction");
        const char * mode = !g.ref_audio.empty() || !g.ref_codes.empty() ? (sent_ins ? "direction" : "clone") : "design";
        printf("gen  %s, %d chars, cfg %.1f, seed %d\n", mode, (int) g.text.size(), g.cfg_scale, g.seed);
        fflush(stdout);

        if (!stream) {
            g.chunk_first = 0;
            std::vector<float> all_samples;
            GenTimings tm;
            const auto t0 = std::chrono::steady_clock::now();
            const auto elapsed = [&] {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            };
            try {
                generate(model, codec, g, [&](const float * s, int n) {
                    all_samples.insert(all_samples.end(), s, s + n);
                    return true;
                }, &tm);
            } catch (const std::exception & e) {
                fprintf(stderr, "\ngeneration error: %s\n", e.what());
                res.status = 500;
                res.set_content("{\"error\":\"generation failed\"}", "application/json");
                return;
            }
            const double secs = (double) all_samples.size() / sr, wall = elapsed();
            printf("gen complete: %.1fs audio in %.2fs [%.1f fps, %.2fx]\n",
                   secs, wall, wall > 0 ? secs * 12.5 / wall : 0.0, wall > 0 ? secs / wall : 0.0);
            fflush(stdout);

            std::vector<uint8_t> pcm = to_pcm16(all_samples.data(), (int) all_samples.size());
            if (fmt == "pcm") {
                res.set_content((const char *) pcm.data(), pcm.size(), "audio/pcm");
            } else {
                std::vector<uint8_t> wav;
                uint32_t data_len = (uint32_t) pcm.size();
                uint32_t riff = 36 + data_len;
                wav.resize(44 + pcm.size());
                uint8_t * p = wav.data();
                std::memcpy(p, "RIFF", 4);
                p[4] = (uint8_t) (riff & 0xff); p[5] = (uint8_t) ((riff >> 8) & 0xff);
                p[6] = (uint8_t) ((riff >> 16) & 0xff); p[7] = (uint8_t) ((riff >> 24) & 0xff);
                std::memcpy(p + 8, "WAVEfmt ", 8);
                p[16] = 16; p[17] = 0; p[18] = 0; p[19] = 0;
                p[20] = 1; p[21] = 0;
                p[22] = 1; p[23] = 0;
                p[24] = (uint8_t) (sr & 0xff); p[25] = (uint8_t) ((sr >> 8) & 0xff);
                p[26] = (uint8_t) ((sr >> 16) & 0xff); p[27] = (uint8_t) ((sr >> 24) & 0xff);
                uint32_t byte_rate = sr * 2;
                p[28] = (uint8_t) (byte_rate & 0xff); p[29] = (uint8_t) ((byte_rate >> 8) & 0xff);
                p[30] = (uint8_t) ((byte_rate >> 16) & 0xff); p[31] = (uint8_t) ((byte_rate >> 24) & 0xff);
                p[32] = 2; p[33] = 0;
                p[34] = 16; p[35] = 0;
                std::memcpy(p + 36, "data", 4);
                p[40] = (uint8_t) (data_len & 0xff); p[41] = (uint8_t) ((data_len >> 8) & 0xff);
                p[42] = (uint8_t) ((data_len >> 16) & 0xff); p[43] = (uint8_t) ((data_len >> 24) & 0xff);
                std::memcpy(p + 44, pcm.data(), pcm.size());
                res.set_content((const char *) wav.data(), wav.size(), "audio/wav");
            }
            return;
        }

        const std::string content_type = fmt == "pcm" ? "audio/pcm" : "audio/wav";
        res.set_chunked_content_provider(
            content_type.c_str(),
            [&model, &codec, g, lock, sr, fmt, verbose = opts.verbose](size_t, httplib::DataSink & sink) {
                if (fmt != "pcm") {
                    // Emit streaming WAV header with large unknown length so streaming audio decoders start immediately
                    std::vector<uint8_t> wav(44);
                    uint8_t * p = wav.data();
                    std::memcpy(p, "RIFF", 4);
                    uint32_t riff = 0x7ffffff0;
                    p[4] = (uint8_t) (riff & 0xff); p[5] = (uint8_t) ((riff >> 8) & 0xff);
                    p[6] = (uint8_t) ((riff >> 16) & 0xff); p[7] = (uint8_t) ((riff >> 24) & 0xff);
                    std::memcpy(p + 8, "WAVEfmt ", 8);
                    p[16] = 16; p[17] = 0; p[18] = 0; p[19] = 0;
                    p[20] = 1; p[21] = 0;
                    p[22] = 1; p[23] = 0;
                    p[24] = (uint8_t) (sr & 0xff); p[25] = (uint8_t) ((sr >> 8) & 0xff);
                    p[26] = (uint8_t) ((sr >> 16) & 0xff); p[27] = (uint8_t) ((sr >> 24) & 0xff);
                    uint32_t byte_rate = sr * 2;
                    p[28] = (uint8_t) (byte_rate & 0xff); p[29] = (uint8_t) ((byte_rate >> 8) & 0xff);
                    p[30] = (uint8_t) ((byte_rate >> 16) & 0xff); p[31] = (uint8_t) ((byte_rate >> 24) & 0xff);
                    p[32] = 2; p[33] = 0;
                    p[34] = 16; p[35] = 0;
                    std::memcpy(p + 36, "data", 4);
                    uint32_t data_len = 0x7ffffff0 - 36;
                    p[40] = (uint8_t) (data_len & 0xff); p[41] = (uint8_t) ((data_len >> 8) & 0xff);
                    p[42] = (uint8_t) ((data_len >> 16) & 0xff); p[43] = (uint8_t) ((data_len >> 24) & 0xff);
                    if (!sink.write((const char *) wav.data(), wav.size())) return false;
                }
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
        if (!req.has_file("audio")) {
            res.status = 400;
            res.set_content("{\"error\":\"audio file is required\"}", "application/json");
            return;
        }
        const auto & af = req.get_file_value("audio");
        std::vector<float> src_audio;
        if (!read_wav_buffer((const uint8_t *) af.content.data(), af.content.size(), sr, src_audio)) {
            res.status = 400;
            res.set_content("{\"error\":\"could not parse audio as wav\"}", "application/json");
            return;
        }
        std::vector<float> ref_audio;
        std::string ref_text = field(req, "ref_text", "");
        if (req.has_file("ref_audio")) {
            const auto & rf = req.get_file_value("ref_audio");
            if (!rf.content.empty())
                read_wav_buffer((const uint8_t *) rf.content.data(), rf.content.size(), sr, ref_audio);
        }
        const std::string vid = field(req, "voice_id", "");
        ConvertOptions opt;
        opt.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
        opt.temperature = (float) atof(field(req, "temperature", "0.6").c_str());
        opt.top_k = atoi(field(req, "top_k", "50").c_str());
        opt.seed = atoi(field(req, "seed", "42").c_str());
        opt.feed_source = field(req, "feed_source", "false") == "true";
        opt.keep_acoustic = atoi(field(req, "keep_acoustic", "0").c_str());
        if (!vid.empty() && !store.take(vid, opt.ref_codes, opt.ref_frames, ref_text)) {
            res.status = 404;
            res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
            return;
        }
        if (ref_audio.empty() && opt.ref_codes.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_audio or a valid voice_id is required\"}", "application/json");
            return;
        }
        int src_T = 0;
        std::vector<int> src_codes = codec.encode(src_audio, src_T);
        printf("convert %d frames (%.1fs), cfg %.1f, seed %d\n", src_T, (double) src_T / 12.5, opt.cfg_scale, opt.seed);
        fflush(stdout);

        std::vector<float> audio = convert_voice(model, codec, src_codes, src_T, ref_audio, ref_text, opt);
        std::vector<uint8_t> pcm = to_pcm16(audio.data(), (int) audio.size());
        res.set_header("X-Sample-Rate", std::to_string(sr));
        res.set_header("X-Sample-Format", "s16le");
        res.set_content((const char *) pcm.data(), pcm.size(), "audio/pcm");
    });

    if (opts.webui) {
        svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
            res.set_content((const char *) breeze_webui::index_html, strlen(breeze_webui::index_html), "text/html");
        });
        svr.Get("/style.css", [](const httplib::Request &, httplib::Response & res) {
            res.set_content((const char *) breeze_webui::style_css, strlen(breeze_webui::style_css), "text/css");
        });
        svr.Get("/app.js", [](const httplib::Request &, httplib::Response & res) {
            res.set_content((const char *) breeze_webui::app_js, strlen(breeze_webui::app_js), "application/javascript");
        });
    }

    printf("listening on http://%s:%d\n", opts.host.c_str(), opts.port);
    svr.listen(opts.host.c_str(), opts.port);
    return 0;
}

} // namespace breeze
