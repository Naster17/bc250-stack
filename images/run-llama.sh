#!/bin/sh
set -eu

profile_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "$profile_dir/config.env"

model=${1:?usage: $0 /absolute/path/to/model.gguf [llama-cli arguments...]}
shift

test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
test -f "$model"

mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm \
    --device /dev/kfd \
    --device /dev/dri/renderD128 \
    --security-opt seccomp=unconfined \
    --network none \
    --volume "$model:/models/model.gguf:ro" \
    --env HSA_ENABLE_SDMA=0 \
    --env GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 \
    --env LD_LIBRARY_PATH="$LLAMA_RUNTIME_DIR:/opt/bc250/hipblas-gfx1013/lib:/opt/bc250/rocblas-gfx1013/lib:/opt/rocm/lib" \
    --env ROCBLAS_TENSILE_LIBPATH=/opt/bc250/rocblas-gfx1013/lib/rocblas/library \
    "$RUNTIME_IMAGE" \
    "$LLAMA_RUNTIME_DIR/llama-cli" \
    --model /models/model.gguf \
    --gpu-layers all \
    --fit off \
    --flash-attn on \
    --ctx-size 128 \
    --batch-size 32 \
    --ubatch-size 32 \
    --no-warmup \
    --no-mmproj \
    "$@"
