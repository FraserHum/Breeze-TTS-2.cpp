#include "ws.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
#define BRZ_BAD INVALID_SOCKET
#define brz_close closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using SOCKET = int;
#define BRZ_BAD (-1)
#define brz_close ::close
#endif

namespace breeze {

// a frame length is attacker controlled, so cap what one message can allocate
static const size_t MAX_MESSAGE = 8u << 20;

static void sha1(const uint8_t * data, size_t len, uint8_t out[20]) {
    uint32_t h[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    std::vector<uint8_t> msg(data, data + len);
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    const uint64_t bits = (uint64_t) len * 8;
    for (int i = 7; i >= 0; i--) msg.push_back((uint8_t) (bits >> (i * 8)));

    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            const uint8_t * p = &msg[off + i * 4];
            w[i] = (uint32_t) p[0] << 24 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 8 | p[3];
        }
        for (int i = 16; i < 80; i++) {
            const uint32_t v = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
            w[i] = v << 1 | v >> 31;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | (~b & d);            k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;                     k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d);   k = 0x8F1BBCDC; }
            else             { f = b ^ c ^ d;                     k = 0xCA62C1D6; }
            const uint32_t t = (a << 5 | a >> 27) + f + e + k + w[i];
            e = d; d = c; c = b << 30 | b >> 2; b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 4; j++) out[i * 4 + j] = (uint8_t) (h[i] >> (24 - j * 8));
}

static std::string base64(const uint8_t * p, size_t n) {
    static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t) p[i] << 16 | (i + 1 < n ? p[i + 1] << 8 : 0) |
                           (i + 2 < n ? p[i + 2] : 0);
        o += T[v >> 18 & 63];
        o += T[v >> 12 & 63];
        o += i + 1 < n ? T[v >> 6 & 63] : '=';
        o += i + 2 < n ? T[v & 63] : '=';
    }
    return o;
}

WsConn::~WsConn() { close(); }

void WsConn::close() {
    if (m_closed.exchange(true)) return;
    brz_close((SOCKET) m_sock);
}

bool WsConn::read_n(void * buf, size_t n) {
    uint8_t * p = (uint8_t *) buf;
    while (n) {
        const int got = ::recv((SOCKET) m_sock, (char *) p, (int) n, 0);
        if (got <= 0) return false;
        p += got;
        n -= (size_t) got;
    }
    return true;
}

bool WsConn::send(int opcode, const void * data, size_t n) {
    if (m_closed) return false;
    std::vector<uint8_t> f;
    f.push_back((uint8_t) (0x80 | opcode));
    if (n < 126) {
        f.push_back((uint8_t) n);
    } else if (n <= 0xFFFF) {
        f.push_back(126);
        f.push_back((uint8_t) (n >> 8));
        f.push_back((uint8_t) n);
    } else {
        f.push_back(127);
        for (int i = 7; i >= 0; i--) f.push_back((uint8_t) ((uint64_t) n >> (i * 8)));
    }
    f.insert(f.end(), (const uint8_t *) data, (const uint8_t *) data + n);

    std::lock_guard<std::mutex> lock(m_send);
    size_t off = 0;
    while (off < f.size()) {
        const int put = ::send((SOCKET) m_sock, (const char *) f.data() + off, (int) (f.size() - off), 0);
        if (put <= 0) { m_closed = true; return false; }
        off += (size_t) put;
    }
    return true;
}

bool WsConn::recv(std::string & out, bool & binary) {
    out.clear();
    binary = false;
    int msg_op = 0;

    for (;;) {
        uint8_t head[2];
        if (!read_n(head, 2)) return false;
        const bool fin = (head[0] & 0x80) != 0;
        const int op = head[0] & 0x0F;
        const bool masked = (head[1] & 0x80) != 0;
        uint64_t len = head[1] & 0x7F;

        if (len == 126) {
            uint8_t e[2];
            if (!read_n(e, 2)) return false;
            len = (uint64_t) e[0] << 8 | e[1];
        } else if (len == 127) {
            uint8_t e[8];
            if (!read_n(e, 8)) return false;
            len = 0;
            for (int i = 0; i < 8; i++) len = len << 8 | e[i];
        }
        if (len > MAX_MESSAGE || out.size() + len > MAX_MESSAGE) return false;

        uint8_t key[4] = { 0, 0, 0, 0 };
        if (masked && !read_n(key, 4)) return false;

        std::string payload;
        payload.resize((size_t) len);
        if (len && !read_n(&payload[0], (size_t) len)) return false;
        if (masked)
            for (size_t i = 0; i < payload.size(); i++) payload[i] ^= (char) key[i & 3];

        if (op == 8) return false;                       // close
        if (op == 9) { send(10, payload.data(), payload.size()); continue; } // ping
        if (op == 10) continue;                          // pong

        if (op != 0) msg_op = op;
        out += payload;
        if (fin) {
            binary = msg_op == 2;
            return true;
        }
    }
}

WsServer::~WsServer() { stop(); }

bool WsServer::start(const std::string & host, int port, Handler h) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    SOCKET s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == BRZ_BAD) return false;
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *) &yes, sizeof yes);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) addr.sin_addr.s_addr = INADDR_ANY;

    if (::bind(s, (sockaddr *) &addr, sizeof addr) != 0 || ::listen(s, 8) != 0) {
        brz_close(s);
        return false;
    }
    m_listen = (uint64_t) s;
    m_handler = std::move(h);
    m_running = true;
    m_thread = std::thread([this] { accept_loop(); });
    return true;
}

void WsServer::stop() {
    if (!m_running.exchange(false)) return;
    brz_close((SOCKET) m_listen);
    if (m_thread.joinable()) m_thread.join();
}

// reads the upgrade request and answers it, leaving the socket talking websocket
static bool handshake(SOCKET s) {
    std::string req;
    char buf[1024];
    while (req.find("\r\n\r\n") == std::string::npos) {
        const int got = ::recv(s, buf, sizeof buf, 0);
        if (got <= 0 || req.size() > 16384) return false;
        req.append(buf, (size_t) got);
    }
    // header names are case insensitive, and browsers do not agree on the casing
    std::string lower = req;
    for (char & c : lower) c = (char) tolower((unsigned char) c);
    const size_t at = lower.find("sec-websocket-key:");
    if (at == std::string::npos) return false;
    size_t b = req.find_first_not_of(" \t", at + 18);
    size_t e = req.find("\r\n", b);
    if (b == std::string::npos || e == std::string::npos) return false;

    const std::string accept = req.substr(b, e - b) + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    uint8_t digest[20];
    sha1((const uint8_t *) accept.data(), accept.size(), digest);

    const std::string res = "HTTP/1.1 101 Switching Protocols\r\n"
                            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                            "Sec-WebSocket-Accept: " + base64(digest, 20) + "\r\n\r\n";
    return ::send(s, res.data(), (int) res.size(), 0) == (int) res.size();
}

void WsServer::accept_loop() {
    while (m_running) {
        sockaddr_in peer{};
        socklen_t plen = sizeof peer;
        SOCKET c = ::accept((SOCKET) m_listen, (sockaddr *) &peer, &plen);
        if (c == BRZ_BAD) break;
        int yes = 1;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char *) &yes, sizeof yes);
        std::thread([this, c] {
            if (handshake(c)) {
                WsConn conn((uint64_t) c);
                m_handler(conn);
            } else {
                brz_close(c);
            }
        }).detach();
    }
}

} // namespace breeze
