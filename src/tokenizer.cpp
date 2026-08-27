#include "breeze/tokenizer.h"
#include "breeze/gguf_loader.h"

#include <algorithm>
#include <limits>

namespace breeze {

static const char * kMeta = "\xE2\x96\x81"; // U+2581 lower one eighth block

static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

bool Tokenizer::load(const GGUFModel & gg) {
    id_to_token = gg.kv_str_array("tokenizer.ggml.tokens");
    if (id_to_token.empty()) return false;
    std::vector<int> types = gg.kv_i32_array("tokenizer.ggml.token_type");
    for (int i = 0; i < (int) id_to_token.size(); i++) {
        token_to_id[id_to_token[i]] = i;
        const std::string & t = id_to_token[i];
        if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>') {
            byte_to_id[std::stoi(t.substr(3, 2), nullptr, 16)] = i;
        }
        if (i < (int) types.size() && (types[i] == 3 || types[i] == 4)) {
            specials.emplace_back(t, i);
        }
    }
    std::vector<std::string> merges = gg.kv_str_array("tokenizer.ggml.merges");
    for (int i = 0; i < (int) merges.size(); i++) merge_rank[merges[i]] = i;
    std::sort(specials.begin(), specials.end(),
              [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    bos_id = gg.kv_u32("tokenizer.ggml.bos_token_id", 2);
    eos_id = gg.kv_u32("tokenizer.ggml.eos_token_id", 1);
    pad_id = gg.kv_u32("tokenizer.ggml.padding_token_id", 0);
    unk_id = gg.kv_u32("tokenizer.ggml.unknown_token_id", 3);
    return true;
}

int Tokenizer::token_id(const std::string & s) const {
    auto it = token_to_id.find(s);
    return it == token_to_id.end() ? -1 : it->second;
}

void Tokenizer::bpe(const std::string & word, std::vector<int> & out) const {
    std::vector<std::string> syms;
    for (size_t i = 0; i < word.size();) {
        int l = utf8_len((unsigned char) word[i]);
        std::string ch = word.substr(i, l);
        i += l;
        if (token_to_id.count(ch)) {
            syms.push_back(ch);
        } else {
            for (char b : ch) {
                auto it = byte_to_id.find((unsigned char) b);
                syms.push_back(it != byte_to_id.end() ? id_to_token[it->second] : ch);
            }
        }
    }
    while (syms.size() > 1) {
        int best = std::numeric_limits<int>::max();
        int best_i = -1;
        for (size_t i = 0; i + 1 < syms.size(); i++) {
            auto it = merge_rank.find(syms[i] + " " + syms[i + 1]);
            if (it != merge_rank.end() && it->second < best) {
                best = it->second;
                best_i = (int) i;
            }
        }
        if (best_i < 0) break;
        syms[best_i] += syms[best_i + 1];
        syms.erase(syms.begin() + best_i + 1);
    }
    for (const std::string & s : syms) {
        auto it = token_to_id.find(s);
        if (it != token_to_id.end()) {
            out.push_back(it->second);
        } else {
            for (char b : s) {
                auto bit = byte_to_id.find((unsigned char) b);
                out.push_back(bit != byte_to_id.end() ? bit->second : unk_id);
            }
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string & text, bool add_bos) const {
    std::vector<int> out;
    if (add_bos) out.push_back(bos_id);
    std::string normal;
    auto flush = [&]() {
        if (normal.empty()) return;
        std::string norm;
        for (size_t i = 0; i < normal.size(); i++) {
            if (normal[i] == ' ') norm += kMeta;
            else norm += normal[i];
        }
        bpe(norm, out);
        normal.clear();
    };
    for (size_t i = 0; i < text.size();) {
        bool matched = false;
        for (const auto & sp : specials) {
            const std::string & c = sp.first;
            if (i + c.size() <= text.size() && text.compare(i, c.size(), c) == 0) {
                flush();
                out.push_back(sp.second);
                i += c.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            normal += text[i];
            i++;
        }
    }
    flush();
    return out;
}

std::string Tokenizer::decode(const std::vector<int> & ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int) id_to_token.size()) continue;
        const std::string & t = id_to_token[id];
        if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>') {
            out += (char) std::stoi(t.substr(3, 2), nullptr, 16);
        } else {
            std::string s = t;
            size_t p;
            while ((p = s.find(kMeta)) != std::string::npos) s.replace(p, 3, " ");
            out += s;
        }
    }
    return out;
}

}
