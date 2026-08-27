# Server API

`breeze-server` is a single binary HTTP server built on cpp-httplib. It loads
one model, serves it over HTTP, and streams raw PCM as the audio is generated
rather than waiting for the whole clip.

## Starting the server

```
breeze-server <model.gguf> [--host H] [--port P] [--webui] [--cpu]
                           [--chunk-first N] [--chunk-max N]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Interface to bind. Use `0.0.0.0` to accept remote connections. |
| `--port` | `8080` | TCP port. |
| `--webui` | off | Also serve the browser UI at `/`. |
| `--cpu` | off | Force the CPU backend instead of Vulkan. |
| `--chunk-first` | `4` | Frames in the first streamed chunk. |
| `--chunk-max` | `25` | Frames the chunk ramps up to. |

```
breeze-server breeze-tts-2-q4_k.gguf --port 8137 --webui
```

### Tuning the chunk ramp

Audio is vocoded in chunks of whole frames, 12.5 frames per second. The first
chunk sets how long the client waits for sound, and every flush pays a fixed
overhead, so the chunk grows by a third each time until it reaches `--chunk-max`.

Lower `--chunk-first` for a faster start. Four frames is about 320 ms of audio
and lands near 350 ms on an RTX 3060; one frame gets there in roughly 220 ms but
flushes far more often.

Raise `--chunk-max` if playback stutters. Larger chunks cut the per flush
overhead and buy the client a deeper queue, at the cost of a coarser stream.
Setting both flags to the same value disables the ramp and streams a fixed size.

The margin you are tuning against is the real time factor. Generating a clip
1.2x faster than it plays leaves very little slack, so a slower quantisation or a
busy GPU will stutter where a faster one does not.

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

Send text fields with `--form-string`, not `-F`. `curl` reads a value that starts
with `(` as the opening of a multipart group, so `-F "text=(sigh) ..."` posts an
empty field and the model ends up reading the multipart boundary out loud.

Voice design:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=Welcome aboard. Your journey begins now." \
  --form-string "instruction=A warm, thoughtful young woman with a clear, calm delivery." \
  --form-string "cfg_scale=1" \
  --form-string "seed=42" \
  -o speech.pcm
```

Voice clone:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=It is good to hear your voice again." \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=This is the exact transcript of the reference audio." \
  -o clone.pcm
```

Voice direction, which is a clone plus an instruction:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=We need to discuss what happened last night." \
  --form-string "instruction=Speak slowly with a restrained, serious tone." \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=This is the exact transcript of the reference audio." \
  --form-string "cfg_scale=1" \
  -o direction.pcm
```

Play the raw stream straight out of `curl`:

```
curl -sN -X POST http://127.0.0.1:8137/v1/audio/speech --form-string "text=Hello there." \
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
`/v1/audio/speech` endpoint.

**Stream while generating** is on by default. The UI reads the response body
incrementally and schedules each PCM chunk through the Web Audio API as it
arrives, so playback starts roughly 2 seconds in rather than after the whole
clip is rendered. A running counter shows how much audio has been produced.
When generation finishes the same PCM is wrapped into a WAV blob for the player
and the download link.

Turn the toggle off to buffer the whole response first and play it back from the
`<audio>` element, which is the better choice if you want to scrub the result
rather than hear it as early as possible.

The assets are compiled into the binary at build time from the
[webui](../webui) folder by `cmake/embed_webui.cmake`, so there is no static
directory to deploy. Editing a file there and rebuilding is enough to pick up
changes.
