# Dev pod (breezetts-dev) — sync, build, verify

`breezetts-dev` (namespace `hermes-voice`, node **2bee**, AMD 780M RADV) is a
mirror of the local Breeze repo. `scripts/pod-sync.sh <branch>` is the only
change channel: the pod receives commit state only (a detached worktree of the
branch tip) — uncommitted local changes never ship.

## What lives where

| Path | Content | Persistence |
|---|---|---|
| `/src` | synced branch tree + vk shim | container layer — wiped on pod recreation |
| `/src/build` | CMake build (VULKAN=ON, Release) | container layer |
| `/models` | `breezetts-cache` PVC (read-only): the gguf models | persists |
| `/cache` | test refs (`calliope.{wav,txt}`, `rt_text.txt`, `voices/`) + run outputs | container layer |

The pod is a standalone privileged pod (node-pinned, `/dev/dri` hostPath,
sleep-keepalive command overriding the image entrypoint); work happens via
`kubectl exec`. The shell inside is **dash** — no bashisms in `sh -c`.
Back up `/cache` before any pod recreation (last backup:
`/tmp/breeze-rt/pod-cache-backup-20260831.tar`).

## The vk shim

Mac source selects the GPU backend with the generic
`ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)`; the pod needs
`ggml_backend_vk_init(0)` (RADV). `pod-vk-shim.patch` (this directory) carries
exactly those two lines in `src/common.cpp`; `pod-sync.sh` applies it after
every sync and aborts the run if it fails to apply. It was extracted from the
pod baseline `1983a00` (= main `0cbfc84` + exactly these two lines, everything
else byte-identical). If main later edits `src/common.cpp` near those lines,
regenerate the patch by diffing main's `common.cpp` against the pod's.

## Baseline gates (main @ 0cbfc84)

| Check | Value |
|---|---|
| binary md5 (`/src/build/breeze-cli`) | `661ef88755d5ec4f3268d13a80ed5626` |
| gate (a) no-ref: `--text "Testing one two. The deploy is live." --seed 42` → `/cache/gate_a.wav` | `619999cceacc6598afebdb9798002bec` |
| gate (b) ref-audio (handoff §3 invocation, `--seed 42 --max-new 120`) → `/cache/gate_b.wav` | `77d514c2be42a273ea7e9926866aebf4` |

`scripts/pod-restore.sh` = `pod-sync main` + all three checks. Every measured
number must cite the binary md5; if any check fails, no number is comparable.
Re-record the constants when main moves.

## Pod recreation (migration checklist)

1. Back up `/cache` (`tar -cf -` over kubectl exec) — container layer.
2. New image tag with `rsync` added (beehive `services/breezetts/Containerfile.dev`).
3. Recreate the pod from its last-applied manifest with the new image tag
   (keep: privileged, `/dev/dri`, models PVC, nodeSelector 2bee, sleep keepalive).
4. `rm -rf /src/.git` — the image bakes a stale clone of the old baseline; the
   pod no longer keeps a git repo, the local repo is the source of truth.
5. Restore `/cache` from the backup.
6. `scripts/pod-sync.sh main` (fresh build dir → script configures it), then
   `scripts/pod-restore.sh` — all checks must PASS.

## Gotchas

- One run at a time: the prod pod (`breezetts-*`) shares the 780M GPU.
- Never touch `third_party/ggml`; never clobber `/cache/calliope.{wav,txt}`,
  `/cache/rt_text.txt`, or `/cache/voices/`.
- The image's baked `/src` + `/src/build` are a stale baseline (old
  `BREEZE_REVISION`); always `pod-sync` before using a recreated pod.
