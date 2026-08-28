#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace breeze {

// just enough of RFC 6455 to stream audio. httplib has no websocket support and does not hand out
// the socket after an upgrade, so this listens on its own port
class WsConn {
public:
    explicit WsConn(uint64_t sock) : m_sock(sock) {}
    ~WsConn();

    // blocks for one whole message. false when the peer closed or the frame was malformed.
    // binary is set when the client sent a binary frame, ping and pong are handled internally
    bool recv(std::string & out, bool & binary);

    bool send_text(const std::string & s) { return send(1, s.data(), s.size()); }
    bool send_binary(const void * p, size_t n) { return send(2, p, n); }

    void close();
    bool alive() const { return !m_closed; }

private:
    bool send(int opcode, const void * data, size_t n);
    bool read_n(void * buf, size_t n);

    uint64_t m_sock;
    std::mutex m_send; // audio goes out from the speaker thread while the reader may send events
    std::atomic<bool> m_closed{false};
};

// one thread per connection, handler returns when the connection is done
class WsServer {
public:
    using Handler = std::function<void(WsConn &)>;

    ~WsServer();
    bool start(const std::string & host, int port, Handler h);
    void stop();

private:
    void accept_loop();

    uint64_t m_listen = 0;
    Handler m_handler;
    std::thread m_thread;
    std::atomic<bool> m_running{false};
};

} // namespace breeze
