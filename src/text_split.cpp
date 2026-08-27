#include "breeze/generation.h"

#include <cstring>

namespace breeze {

static size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

// cjk characters are whole syllables, they take far longer to speak than one latin letter
static int weigh(const std::string & s) {
    int w = 0;
    for (size_t i = 0; i < s.size(); ) {
        const size_t n = utf8_len((unsigned char) s[i]);
        w += n == 1 ? 1 : 3;
        i += n;
    }
    return w;
}

static bool is_cjk_stop(const std::string & c) {
    static const char * stops[] = { "。", "！", "？", "；", "…", "\xef\xbc\x8e" };
    for (const char * s : stops) if (c == s) return true;
    return false;
}

static std::vector<std::string> split_sentences(const std::string & t) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < t.size(); ) {
        const size_t n = utf8_len((unsigned char) t[i]);
        const std::string ch = t.substr(i, n);
        cur += ch;
        i += n;
        bool brk = false;
        if (n == 1) {
            const char c = ch[0];
            if (c == '\n') {
                brk = true;
            } else if (c == '.' || c == '!' || c == '?' || c == ';') {
                // only a real break when a gap follows, so 3.14 and Dr. stay in one piece
                size_t j = i;
                while (j < t.size() && strchr("\"')]", t[j])) j++;
                brk = j >= t.size() || t[j] == ' ' || t[j] == '\n';
                if (brk) { cur += t.substr(i, j - i); i = j; }
            }
        } else {
            brk = is_cjk_stop(ch);
        }
        if (brk) { out.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// a single sentence over budget still has to go somewhere, so break it at clause boundaries
static std::vector<std::string> split_clauses(const std::string & s, int budget) {
    if (weigh(s) <= budget) return { s };
    std::vector<std::string> out;
    std::string cur;
    int cw = 0, since_break = 0;
    for (size_t i = 0; i < s.size(); ) {
        const size_t n = utf8_len((unsigned char) s[i]);
        const std::string ch = s.substr(i, n);
        cur += ch;
        cw += n == 1 ? 1 : 3;
        since_break += n == 1 ? 1 : 3;
        i += n;
        const bool comma = ch == "," || ch == "，" || ch == "、" || ch == ":";
        const bool space = ch == " ";
        if (cw >= budget && (comma || (space && since_break > budget / 4))) {
            out.push_back(cur);
            cur.clear();
            cw = 0;
            since_break = 0;
        } else if (comma || space) {
            since_break = 0;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<std::string> split_text(const std::string & text, int budget, int first_budget) {
    if (budget <= 0 || weigh(text) <= budget) return { text };
    std::vector<std::string> out;
    std::string cur;
    int cw = 0;
    int limit = first_budget > 0 ? first_budget : budget;
    for (const std::string & s : split_sentences(text)) {
        for (const std::string & p : split_clauses(s, budget)) {
            const int w = weigh(p);
            if (cw > 0 && cw + w > limit) {
                out.push_back(cur);
                cur.clear();
                cw = 0;
                limit = budget;
            }
            cur += p;
            cw += w;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    if (out.empty()) out.push_back(text);
    return out;
}

}
