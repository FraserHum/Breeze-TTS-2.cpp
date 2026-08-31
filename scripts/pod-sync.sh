#!/bin/sh
# pod-sync: sync a Breeze branch from the local repo into the dev pod,
# apply the vk deploy shim, rebuild, and print a state receipt.
#
#   usage: scripts/pod-sync.sh [branch] [--no-build]
#
#   branch      branch to sync (default: current branch)
#   --no-build  sync + shim only, skip the pod build
#
# The pod receives commit state only: the branch tip is checked out into a
# detached worktree (/tmp/breeze-pod-sync/work) and that tree is what ships.
# Uncommitted local changes never reach the pod.
#
# Never touched: /src/.git, /src/build, /models, /cache.
# The pod shell is dash; git runs only on the Mac side.
set -eu

NS=hermes-voice
POD=${BREEZE_DEV_POD:-breezetts-dev}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORKTREE=/tmp/breeze-pod-sync/work
LOCK=/tmp/breeze-pod-sync.lock

BRANCH=""
DO_BUILD=1
for a in "$@"; do
    case "$a" in
        --no-build) DO_BUILD=0 ;;
        -*) echo "pod-sync: unknown flag: $a" >&2; exit 2 ;;
        *) BRANCH=$a ;;
    esac
done
[ -n "$BRANCH" ] || BRANCH=$(git -C "$REPO_DIR" branch --show-current)
[ -n "$BRANCH" ] || { echo "pod-sync: pass a branch name" >&2; exit 2; }

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "pod-sync: $LOCK held (another sync running, or stale lock from a crash)" >&2
    exit 1
fi
trap 'rmdir "$LOCK"' EXIT

SHA=$(git -C "$REPO_DIR" rev-parse --verify "refs/heads/$BRANCH^{commit}")

# 1. branch tip into a detached worktree (commit state only)
if [ ! -e "$WORKTREE/.git" ]; then
    git -C "$REPO_DIR" worktree add --detach "$WORKTREE" "$SHA"
fi
git -C "$WORKTREE" checkout --detach -q "$SHA"
git -C "$WORKTREE" submodule update --init -q   # pinned ggml, pristine

# 2. sync the tree into /src: rsync if the pod has it, else tar + file manifest
EXCLUDES="--exclude .git --exclude build"
if kubectl exec -n "$NS" "$POD" -- sh -c 'command -v rsync >/dev/null 2>&1'; then
    # empty host part: with -e, rsync runs "<cmd> <host> rsync --server ...";
    # a non-empty host here would be exec'd inside the container by kubectl
    rsync -ac --delete $EXCLUDES \
        -e "kubectl exec -i -n $NS $POD --" \
        "$WORKTREE"/ :/src/
    SYNC_MODE=rsync
else
    (cd "$WORKTREE" && find . \( -name .git -o -name build \) -prune -o -type f -print | sort) \
        > /tmp/.pod-sync.local.$$
    kubectl exec -n "$NS" "$POD" -- sh -c 'cd /src && find . \( -name .git -o -name build \) -prune -o -type f -print | sort' \
        > /tmp/.pod-sync.remote.$$
    # delete files that no longer exist locally
    comm -23 /tmp/.pod-sync.remote.$$ /tmp/.pod-sync.local.$$ \
        | kubectl exec -i -n "$NS" "$POD" -- sh -c 'cd /src && while IFS= read -r f; do rm -f "$f"; done'
    (cd "$WORKTREE" && tar -cf - --exclude .git --exclude build .) \
        | kubectl exec -i -n "$NS" "$POD" -- tar -C /src -xf -
    rm -f /tmp/.pod-sync.local.$$ /tmp/.pod-sync.remote.$$
    SYNC_MODE=tar
fi

# 3. vk deploy shim: the sync delivers Mac source, the pod needs vk device init
if kubectl exec -i -n "$NS" "$POD" -- patch -p1 --forward -s -d /src \
        < "$REPO_DIR/docs/deploy/pod-vk-shim.patch"; then
    SHIM=applied
elif kubectl exec -n "$NS" "$POD" -- grep -q 'ggml_backend_vk_init(0)' /src/src/common.cpp; then
    SHIM=already
else
    echo "pod-sync: vk shim failed to apply - aborting (the build would fall back to CPU)" >&2
    exit 1
fi

# 4. stale-mtime guard (tar/rsync can carry old mtimes -> CMake skips recompiles)
kubectl exec -n "$NS" "$POD" -- sh -c \
    'cd /src && find src include apps -type f \( -name "*.cpp" -o -name "*.h" \) -exec touch {} +'

# 5. build (configure first if the build dir is new)
if [ "$DO_BUILD" = 1 ]; then
    if ! kubectl exec -n "$NS" "$POD" -- sh -c '[ -f /src/build/CMakeCache.txt ]'; then
        kubectl exec -n "$NS" "$POD" -- sh -c \
            'cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DBREEZE_VULKAN=ON -DBREEZE_BUILD_SERVER=ON -DBREEZE_BUILD_SHARED=OFF'
    fi
    kubectl exec -n "$NS" "$POD" -- sh -c 'cmake --build /src/build -j"$(nproc)"'
    BUILD=ok
else
    BUILD=skipped
fi

# 6. receipt
MD5=$(kubectl exec -n "$NS" "$POD" -- sh -c 'md5sum /src/build/breeze-cli 2>/dev/null | cut -d" " -f1')
echo "=== pod-sync receipt ==="
echo "branch:  $BRANCH @ $SHA"
echo "sync:    $SYNC_MODE (never touches .git, build, /models, /cache)"
echo "shim:    $SHIM"
echo "build:   $BUILD"
echo "binary:  md5 ${MD5:-none}"
echo "gates:   scripts/pod-restore.sh (binary md5 + both generation gates)"
