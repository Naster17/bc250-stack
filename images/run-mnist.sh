#!/bin/sh
# MNIST on the deployed stack: r1.4 stable container, tuned DB, EPOCHS env.
# Usage: EPOCHS=100 doas ./run-mnist.sh
set -eu
test "$(id -u)" -eq 0 || { echo "run through doas" >&2; exit 1; }
. /home/nik/bc250work/bc250-rocm-profile/config.env
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
podman start "$RUNTIME_CONTAINER" >/dev/null 2>&1 || true
exec podman run --rm --device /dev/kfd --device /dev/dri/renderD128 --security-opt seccomp=unconfined --network none \
    -v /home/nik/bc250work/mnist:/data/mnist:ro \
    -v /home/nik/bc250work/mnist-train.sh:/mnist-train.sh:ro \
    --env HSA_ENABLE_SDMA=1 \
    --env OMP_NUM_THREADS=4 \
    --env ROCBLAS_TENSILE_LIBPATH=/opt/bc250/rocblas-gfx1013/lib/rocblas/library \
    --env MIOPEN_USER_DB_PATH=/opt/bc250/miopen-db \
    --env EPOCHS="${EPOCHS:-5}" \
    "$RUNTIME_IMAGE" /bin/sh /mnist-train.sh
