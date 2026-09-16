#!/bin/sh
set -eu

profile_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "$profile_dir/config.env"

test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}

mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
podman start "$RUNTIME_CONTAINER" >/dev/null 2>&1 || true
exec podman exec \
    --env HSA_ENABLE_SDMA=1 \
    --env OMP_NUM_THREADS=4 \
    --env ROCBLAS_TENSILE_LIBPATH=/opt/bc250/rocblas-gfx1013/lib/rocblas/library \
    "$RUNTIME_CONTAINER" "$TORCH_VENV/bin/python" "$@"
