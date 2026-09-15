#!/usr/bin/env python3
"""Check a running server rejects unsupported formats before synthesis."""
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def check(base_url):
    for response_format in ("mp3", "wav", "pcm", None):
        payload = {"input": ""}
        if response_format is not None:
            payload["response_format"] = response_format
        request = Request(
            base_url.rstrip("/") + "/v1/audio/speech",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            assert response.status == 400, response.status
            body = json.load(response)
        expected = (
            "unsupported response_format; only pcm is supported"
            if response_format in ("mp3", "wav") else "text is required"
        )
        assert body["error"] == expected, (response_format, body)
    print("PASS: unsupported formats rejected; PCM and omitted format accepted")


if __name__ == "__main__":
    check(sys.argv[1])
