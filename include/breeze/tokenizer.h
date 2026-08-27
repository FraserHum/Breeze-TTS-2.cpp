#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace breeze {

struct GGUFModel;

struct Tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> merge_rank;
    std::vector<std::pair<std::string, int>> specials; // content -> id, longest first
    std::unordered_map<int, int> byte_to_id;
    int bos_id = 2, eos_id = 1, pad_id = 0, unk_id = 3;

    bool load(const GGUFModel & gg);
    int token_id(const std::string & s) const;
    std::vector<int> encode(const std::string & text, bool add_bos) const;
    std::string decode(const std::vector<int> & ids) const;

private:
    void bpe(const std::string & word, std::vector<int> & out) const;
};

}
