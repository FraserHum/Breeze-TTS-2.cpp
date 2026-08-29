#pragma once

#include "breeze/codec.h"
#include "breeze/model.h"

#include "httplib.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace breeze {

inline std::string json_extract_field(const std::string & json, const char * key) {
    const std::string pat = "\"" + std::string(key) + "\"";
    size_t at = json.find(pat);
    if (at == std::string::npos) return "";
    at = json.find(':', at + pat.size());
    if (at == std::string::npos) return "";
    at = json.find_first_not_of(" \t\n\r", at + 1);
    if (at == std::string::npos) return "";
    if (json[at] == '"') {
        std::string out;
        for (size_t i = at + 1; i < json.size(); i++) {
            const char c = json[i];
            if (c == '\\' && i + 1 < json.size()) {
                const char n = json[++i];
                if (n == 'n') out += '\n';
                else if (n == 't') out += '\t';
                else if (n == 'r') out += '\r';
                else if (n == '"') out += '"';
                else if (n == '\\') out += '\\';
                else { out += c; out += n; }
            } else if (c == '"') {
                return out;
            } else {
                out += c;
            }
        }
        return out;
    }
    size_t end = json.find_first_of(",}\n\r", at);
    if (end == std::string::npos) end = json.size();
    std::string out = json.substr(at, end - at);
    while (!out.empty() && (out.back() == ' ' || out.back() == '\t' || out.back() == '\r'))
        out.pop_back();
    return out;
}

// form fields arrive as either a file part or a plain param depending on the client,
// or as JSON fields for OpenAI-compatible requests
inline std::string field(const httplib::Request & req, const char * name, const std::string & def) {
    if (req.has_file(name)) return req.get_file_value(name).content;
    if (req.has_param(name)) return req.get_param_value(name);
    if (strcmp(name, "text") == 0) {
        if (req.has_file("input")) return req.get_file_value("input").content;
        if (req.has_param("input")) return req.get_param_value("input");
    } else if (strcmp(name, "voice_id") == 0) {
        if (req.has_file("voice")) return req.get_file_value("voice").content;
        if (req.has_param("voice")) return req.get_param_value("voice");
    } else if (strcmp(name, "instruction") == 0) {
        if (req.has_file("instructions")) return req.get_file_value("instructions").content;
        if (req.has_param("instructions")) return req.get_param_value("instructions");
    }
    if (!req.body.empty()) {
        std::string val = json_extract_field(req.body, name);
        if (!val.empty()) return val;
        if (strcmp(name, "text") == 0) {
            val = json_extract_field(req.body, "input");
            if (!val.empty()) return val;
        } else if (strcmp(name, "voice_id") == 0) {
            val = json_extract_field(req.body, "voice");
            if (!val.empty()) return val;
        } else if (strcmp(name, "instruction") == 0) {
            val = json_extract_field(req.body, "instructions");
            if (!val.empty()) return val;
        }
    }
    return def;
}

struct CachedVoice {
    std::vector<int> codes;
    int frames = 0;
    std::string text;
    double encode_ms = 0;
    bool saved = false; // has a .breeze file behind it, so it survives a restart
};

// reference voices the server already encoded, both the ones posted at runtime and the files on disk
class VoiceStore {
public:
    void load_dir(const std::string & dir, int n_codebooks);

    // fills in a cached reference, false only when the id is unknown
    bool take(const std::string & id, std::vector<int> & codes, int & frames, std::string & text);

    void add_routes(httplib::Server & svr, BreezeModel & model, MimiCodec & codec, std::mutex & gpu,
                    const std::string & dir);

private:
    std::map<std::string, CachedVoice> m_voices;
    std::vector<std::string> m_order;
    std::mutex m_mutex;
    static const size_t max_voices = 64;
};

} // namespace breeze
