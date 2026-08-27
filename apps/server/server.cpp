#include "server.h"

#include "breeze/audio.h"
#include "breeze/generation.h"
#include "breeze/model.h"

#include "httplib.h"
#include "breeze_webui_assets.h"

#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

namespace breeze {

static std::string field(const httplib::Request & req, const char * name, const std::string & def) {
    if (req.has_file(name)) return req.get_file_value(name).content;
    if (req.has_param(name)) return req.get_param_value(name);
    return def;
}

int run_server(const ServerOptions & opts) {
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

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\",\"sample_rate\":" + std::to_string(sr) + "}", "application/json");
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
        g.max_new_tokens = atoi(field(req, "max_new_tokens", "0").c_str());
        g.split_chars = atoi(field(req, "split_chars", "600").c_str());
        g.chunk_first = opts.chunk_first;
        g.chunk_max = opts.chunk_max;
        if (req.has_file("ref_audio")) {
            const auto & f = req.get_file_value("ref_audio");
            if (!f.content.empty())
                read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, g.ref_audio);
        }
        if (g.text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"text is required\"}", "application/json");
            return;
        }

        res.set_header("X-Sample-Rate", std::to_string(sr));
        res.set_header("X-Sample-Format", "s16le");
        res.set_header("Cache-Control", "no-store");
        res.set_chunked_content_provider(
            "audio/pcm",
            [&model, &codec, g, lock](size_t, httplib::DataSink & sink) {
                try {
                    generate(model, codec, g, [&](const float * s, int n) {
                        std::vector<uint8_t> pcm = to_pcm16(s, n);
                        return sink.write((const char *) pcm.data(), pcm.size());
                    });
                } catch (const std::exception & e) {
                    fprintf(stderr, "generation error: %s\n", e.what());
                }
                sink.done();
                return true;
            });
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
