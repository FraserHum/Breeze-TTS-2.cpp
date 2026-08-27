#pragma once

#include <string>

namespace breeze {

struct ServerOptions {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    bool use_gpu = true;
    bool webui = false;
};

int run_server(const ServerOptions & opts);

}
