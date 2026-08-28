#pragma once

#include <string>
#include <vector>

namespace breeze {

// a saved reference voice. the codes are what the codec produced for the clip, so loading one skips
// the encode entirely, which is most of the wait before the first audio comes out
struct Voice {
    std::string name;
    std::string text; // exact transcript of the clip the codes came from
    std::vector<int> codes;
    int frames = 0;
    int n_codebooks = 0;
    int sample_rate = 24000;
};

bool save_voice(const std::string & path, const Voice & v);
bool load_voice(const std::string & path, Voice & v);

// every .breeze in dir, named after the file. missing or unreadable dirs just come back empty
std::vector<Voice> load_voice_dir(const std::string & dir);

// filenames come from requests, so keep them to something that cannot walk out of the folder
bool valid_voice_name(const std::string & name);

} // namespace breeze
