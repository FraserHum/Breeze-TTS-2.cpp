#!/bin/sh
# pod-restore: bring the dev pod back to the known baseline and verify it.
#
#   1. pod-sync main (sync + shim + build)
#   2. binary md5 check
#   3. both generation gates (no-ref + ref-audio)
#
# All three must hold before any measured number is comparable to baseline.
# The gate md5s are for main @ 0cbfc84 - re-record them (and update here)
# when main moves.
set -eu

NS=hermes-voice
POD=${BREEZE_DEV_POD:-breezetts-dev}
DIR=$(cd "$(dirname "$0")/.." && pwd)

BIN_BASE=661ef88755d5ec4f3268d13a80ed5626
GATE_A=619999cceacc6598afebdb9798002bec
GATE_B=77d514c2be42a273ea7e9926866aebf4

sh "$DIR/scripts/pod-sync.sh" main

BIN=$(kubectl exec -n "$NS" "$POD" -- md5sum /src/build/breeze-cli | cut -d' ' -f1)
echo "binary:  $BIN (baseline $BIN_BASE)"
[ "$BIN" = "$BIN_BASE" ] || { echo "FAIL: binary differs from baseline" >&2; exit 1; }

kubectl exec -n "$NS" "$POD" -- sh -c \
    '/src/build/breeze-cli /models/breeze-tts-2-q4_k.gguf --text "Testing one two. The deploy is live." --seed 42 --timings --output /cache/gate_a.wav' >/dev/null
A=$(kubectl exec -n "$NS" "$POD" -- md5sum /cache/gate_a.wav | cut -d' ' -f1)
echo "gate a:  $A (baseline $GATE_A)"
[ "$A" = "$GATE_A" ] || { echo "FAIL: gate (a) no-ref" >&2; exit 1; }

kubectl exec -n "$NS" "$POD" -- sh -c \
    '/src/build/breeze-cli /models/breeze-tts-2-q4_k.gguf --ref-audio /cache/calliope.wav --ref-text "$(cat /cache/calliope.txt)" --text "$(cat /cache/rt_text.txt)" --seed 42 --max-new 120 --timings --output /cache/gate_b.wav' >/dev/null
B=$(kubectl exec -n "$NS" "$POD" -- md5sum /cache/gate_b.wav | cut -d' ' -f1)
echo "gate b:  $B (baseline $GATE_B)"
[ "$B" = "$GATE_B" ] || { echo "FAIL: gate (b) ref-audio" >&2; exit 1; }

echo "=== pod-restore: PASS (binary + both gates match baseline) ==="
