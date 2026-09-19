#include "breeze/voice.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace breeze {

static const char MAGIC[4] = { 'B', 'R', 'Z', 'V' };
static const uint32_t VERSION = 1;

static bool put32(FILE * f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static bool get32(FILE * f, uint32_t & v) { return fread(&v, 4, 1, f) == 1; }

bool valid_voice_name(const std::string & name) {
    if (name.empty() || name.size() > 64) return false;
    for (char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

static bool get32_buf(const uint8_t * & p, const uint8_t * end, uint32_t & v) {
    if (end - p < 4) return false;
    memcpy(&v, p, 4);
    p += 4;
    return true;
}

bool save_voice(const std::string & path, const Voice & v) {
    if (v.frames <= 0 || v.n_codebooks <= 0 ||
        v.codes.size() != (size_t) v.frames * v.n_codebooks) return false;

    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(MAGIC, 1, 4, f) == 4 &&
              put32(f, VERSION) && put32(f, (uint32_t) v.sample_rate) &&
              put32(f, (uint32_t) v.n_codebooks) && put32(f, (uint32_t) v.frames) &&
              put32(f, (uint32_t) v.text.size());
    if (ok && !v.text.empty()) ok = fwrite(v.text.data(), 1, v.text.size(), f) == v.text.size();
    for (size_t i = 0; ok && i < v.codes.size(); i++) ok = put32(f, (uint32_t) (int32_t) v.codes[i]);
    fclose(f);
    if (!ok) std::filesystem::remove(path, ec);
    return ok;
}

bool load_voice(const std::string & path, Voice & v) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;

    char magic[4];
    uint32_t version = 0, sr = 0, ncb = 0, frames = 0, text_len = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, MAGIC, 4) == 0 &&
              get32(f, version) && version == VERSION &&
              get32(f, sr) && get32(f, ncb) && get32(f, frames) && get32(f, text_len);

    // a corrupt header should not turn into a giant allocation
    if (ok) ok = ncb > 0 && ncb <= 64 && frames > 0 && frames <= 100000 && text_len <= (1u << 20);
    if (ok) {
        v.text.resize(text_len);
        if (text_len) ok = fread(&v.text[0], 1, text_len, f) == text_len;
    }
    if (ok) {
        v.codes.resize((size_t) frames * ncb);
        for (size_t i = 0; ok && i < v.codes.size(); i++) {
            uint32_t c = 0;
            ok = get32(f, c);
            v.codes[i] = (int) (int32_t) c;
        }
    }
    fclose(f);
    if (!ok) return false;

    v.sample_rate = (int) sr;
    v.n_codebooks = (int) ncb;
    v.frames = (int) frames;
    v.name = std::filesystem::path(path).stem().string();
    return true;
}

std::vector<Voice> load_voice_dir(const std::string & dir) {
    std::vector<Voice> out;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return out;
    for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec) || e.path().extension() != ".breeze") continue;
        Voice v;
        if (load_voice(e.path().string(), v)) out.push_back(std::move(v));
        else fprintf(stderr, "skipping %s, not a readable voice file\n", e.path().string().c_str());
    }
    return out;
}

bool parse_voice_bytes(const uint8_t * data, size_t size, const std::string & name, Voice & v) {
    if (size < 28 || memcmp(data, MAGIC, 4) != 0) return false;
    const uint8_t * p = data + 4, * end = data + size;
    uint32_t version = 0, sr = 0, ncb = 0, frames = 0, text_len = 0;
    if (!get32_buf(p, end, version) || version != VERSION) return false;
    if (!get32_buf(p, end, sr) || !get32_buf(p, end, ncb) || !get32_buf(p, end, frames) ||
        !get32_buf(p, end, text_len)) return false;

    // same bounds as load_voice so corrupt headers cannot turn into giant allocations
    if (ncb == 0 || ncb > 64 || frames == 0 || frames > 100000 || text_len > (1u << 20)) return false;
    if ((size_t) (end - p) < (size_t) text_len) return false;

    v.text.assign((const char *) p, text_len);
    p += text_len;
    v.codes.resize((size_t) frames * ncb);
    for (size_t i = 0; i < v.codes.size(); i++) {
        uint32_t c = 0;
        if (!get32_buf(p, end, c)) return false;
        v.codes[i] = (int) (int32_t) c;
    }
    if (p != end) return false; // trailing garbage is not a BRZV file

    v.name = name;
    v.sample_rate = (int) sr;
    v.n_codebooks = (int) ncb;
    v.frames = (int) frames;
    return true;
}

std::vector<Voice> load_embedded_voices(const GGUFModel & gg, std::string * default_name) {
    std::vector<Voice> out;
    if (default_name) *default_name = gg.kv_str("breeze.embedded_voice.default", "");
    const std::vector<std::string> names = gg.kv_str_array("breeze.embedded_voice_names");
    for (const std::string & name : names) {
        if (!valid_voice_name(name)) {
            fprintf(stderr, "skipping embedded voice %s, invalid name\n", name.c_str());
            continue;
        }
        const std::string key = "breeze.embedded_voice." + name;
        const std::vector<uint8_t> raw = gg.kv_bytes(key.c_str());
        Voice v;
        if (!parse_voice_bytes(raw.data(), raw.size(), name, v)) {
            fprintf(stderr, "skipping embedded voice %s, unreadable\n", name.c_str());
            continue;
        }
        out.push_back(std::move(v));
    }
    return out;
}

} // namespace breeze
