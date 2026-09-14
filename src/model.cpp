#include "breeze/model.h"

#include <cstdlib>
#include <cstring>

namespace breeze {

bool BreezeModel::load(const std::string & path, bool prefer_gpu) {
    backend.init(prefer_gpu);
    if (!gg.load(path, backend)) {
        fprintf(stderr, "gg.load failed for %s\n", path.c_str());
        return false;
    }
    cfg = parse_config(gg);
    if (!tok.load(gg)) {
        fprintf(stderr, "tok.load failed for %s\n", path.c_str());
        return false;
    }
    const char * env_pack = std::getenv("BREEZE_PACK_WEIGHTS");
    if (!env_pack || std::strcmp(env_pack, "0") != 0) {
        if (!gg.pack_weights(cfg, backend)) {
            fprintf(stderr, "pack_weights failed for %s\n", path.c_str());
        }
    }
    return true;
}

void BreezeModel::free() {
    gg.free();
    backend.free();
}

}
