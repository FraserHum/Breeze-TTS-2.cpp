# Server API

`breeze-server` is a single binary HTTP server built on cpp-httplib. It loads
one model, serves it over HTTP, and streams raw PCM as the audio is generated
rather than waiting for the whole clip.

## Starting the server

```
breeze-server <model.gguf> [--host H] [--port P] [--webui] [--cpu]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Interface to bind. Use `0.0.0.0` to accept remote connections. |
| `--port` | `8080` | TCP port. |
| `--webui` | off | Also serve the browser UI at `/`. |
| `--cpu` | off | Force the CPU backend instead of Vulkan. |

```
breeze-server breeze-tts-2-q4_k.gguf --port 8137 --webui
```

There is no authentication and no rate limiting. Do not expose it directly to
the internet; put it behind a reverse proxy that handles both.

## `GET /health`

Liveness probe. Returns once the model is loaded and ready.

```
curl http://127.0.0.1:8137/health
```

```json
{"status":"ok","sample_rate":24000}
```

The server does not accept connections until loading finishes, so a successful
response also means the model is warm. Poll this after startup instead of
guessing at a delay.

## `POST /v1/audio/speech`

Generates speech and streams it back.

Accepts `multipart/form-data` (needed for the reference audio upload) or
`application/x-www-form-urlencoded`.

### Fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `text` | string | required | Text to speak, UTF-8. |
| `instruction` | string | `Speak clearly and naturally.` | Voice description or delivery direction. |
| `ref_audio` | file | none | Reference WAV for cloning. Any sample rate or channel count; resampled to mono at the model rate. |
| `ref_text` | string | empty | Exact transcript of `ref_audio`. Required whenever `ref_audio` is present. |
| `cfg_scale` | float | `1.0` | Classifier free guidance. `1.0` disables it. |
| `seed` | int | `42` | RNG seed. |

### Response

`200 OK` with a chunked body.

| Header | Value |
| --- | --- |
| `Content-Type` | `audio/pcm` |
| `Transfer-Encoding` | `chunked` |
| `X-Sample-Rate` | `24000` |
| `X-Sample-Format` | `s16le` |
| `Cache-Control` | `no-store` |

The body is **headerless** signed 16 bit little endian mono PCM. It is not a
WAV file. Wrap it yourself if you need one, or use the CLI which writes WAV
directly.

Chunks arrive roughly every 2 seconds of generated audio, so playback can start
long before generation finishes.

### Errors

| Status | Body | Cause |
| --- | --- | --- |
| `400` | `{"error":"text is required"}` | `text` missing or empty. |
| `409` | `{"error":"busy"}` | Another generation is already running. |

The server holds one model and serves one request at a time. A second
concurrent request is rejected immediately with `409` rather than queued, so
clients should retry with backoff or run several servers behind a load
balancer.

Because the response is streamed, a failure that happens **after** the first
chunk cannot change the status code. The connection is closed early instead, so
treat a truncated body as an error.

### Examples

Voice design:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  -F "text=Welcome aboard. Your journey begins now." \
  -F "instruction=A warm, thoughtful young woman with a clear, calm delivery." \
  -F "cfg_scale=1" \
  -F "seed=42" \
  -o speech.pcm
```

Voice clone:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  -F "text=It is good to hear your voice again." \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=This is the exact transcript of the reference audio." \
  -o clone.pcm
```

Voice direction, which is a clone plus an instruction:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  -F "text=We need to discuss what happened last night." \
  -F "instruction=Speak slowly with a restrained, serious tone." \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=This is the exact transcript of the reference audio." \
  -F "cfg_scale=1" \
  -o direction.pcm
```

Play the raw stream straight out of `curl`:

```
curl -sN -X POST http://127.0.0.1:8137/v1/audio/speech -F "text=Hello there." \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

Convert to WAV:

```
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav
```

## Streaming client sketch

```python
import struct
import urllib.request

body, boundary = build_multipart({"text": "Streaming from python."})
req = urllib.request.Request(
    "http://127.0.0.1:8137/v1/audio/speech",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

with urllib.request.urlopen(req) as r:
    rate = int(r.headers["X-Sample-Rate"])
    tail = b""
    while chunk := r.read(8192):
        chunk = tail + chunk
        n = len(chunk) // 2
        tail = chunk[n * 2:]
        samples = struct.unpack(f"<{n}h", chunk[:n * 2])
        play(samples, rate)
```

Buffer any odd trailing byte between reads, since a chunk boundary can land in
the middle of a sample.

## Web UI

`--webui` mounts a dark brutalist single page UI that mirrors the reference
Gradio demo.

| Route | Serves |
| --- | --- |
| `GET /` | `index.html` |
| `GET /style.css` | stylesheet |
| `GET /app.js` | client script |

It has three tabs matching the three generation modes, and it posts to the same
`/v1/audio/speech` endpoint, wraps the returned PCM into a WAV blob in the
browser, then plays it and offers a download.

The assets are compiled into the binary at build time from the
[webui](../webui) folder by `cmake/embed_webui.cmake`, so there is no static
directory to deploy. Editing a file there and rebuilding is enough to pick up
changes.
