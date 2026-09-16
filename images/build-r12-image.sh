#!/bin/sh
set -eu
# r1.2 runtime image: r1.1 payload + full native gfx1013 ROCm stack + torch.
# GPU-free build: no devices mapped. Network allowed for apt/PyPI only.
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
work=/home/nik/bc250work
src_img=localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.1
new_img=localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.2
tmpc=bc250-r12-staging
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
podman rm -f "$tmpc" 2>/dev/null || true
podman create --name "$tmpc" "$src_img" sleep 2147483647
podman start "$tmpc"
copy_prefix() {
    dest="$2"
    podman exec "$tmpc" mkdir -p "$dest"
    rm -rf /tmp/r12copy
    mkdir -p /tmp/r12copy
    cp -a "$1/." /tmp/r12copy/
    podman cp /tmp/r12copy/. "$tmpc:$dest"
    rm -rf /tmp/r12copy
}
copy_prefix $work/rocblas-build-gfx1013/rocblas-prefix /opt/bc250/rocblas-gfx1013
copy_prefix $work/hipblas-build-gfx1013/hipblas-prefix /opt/bc250/hipblas-gfx1013
copy_prefix $work/hipblas-build-gfx1013/hipblas-common-prefix /opt/bc250/hipblas-common
copy_prefix $work/rocfft-build-gfx1013/rocfft-prefix /opt/bc250/rocfft-gfx1013
copy_prefix $work/hipfft-build-gfx1013/hipfft-prefix /opt/bc250/hipfft-gfx1013
copy_prefix $work/rocrand-build-gfx1013/rocrand-prefix /opt/bc250/rocrand-gfx1013
copy_prefix $work/rocrand-build-gfx1013/hiprand-prefix /opt/bc250/hiprand-gfx1013
copy_prefix $work/rocsparse-build-gfx1013/rocsparse-prefix /opt/bc250/rocsparse-gfx1013
copy_prefix $work/rocsparse-build-gfx1013/hipsparse-prefix /opt/bc250/hipsparse-gfx1013
copy_prefix $work/rocsolver-build-gfx1013/rocsolver-prefix /opt/bc250/rocsolver-gfx1013
copy_prefix $work/rocsolver-build-gfx1013/hipsolver-prefix /opt/bc250/hipsolver-gfx1013
copy_prefix $work/hipblaslt-build-gfx1013/hipblaslt-prefix /opt/bc250/hipblaslt-gfx1013
copy_prefix $work/roctracer-build-gfx1013/roctracer-prefix /opt/bc250/roctracer-gfx1013
copy_prefix $work/rocm-headers-gfx1013/rocprim-prefix /opt/bc250/rocprim-gfx1013
copy_prefix $work/rocm-headers-gfx1013/hipcub-prefix /opt/bc250/hipcub-gfx1013
copy_prefix $work/rocm-headers-gfx1013/rocthrust-prefix /opt/bc250/rocthrust-gfx1013
copy_prefix $work/miopen-build-gfx1013/miopen-prefix /opt/bc250/miopen-gfx1013
copy_prefix $work/openmp-prefix /opt/bc250/openmp-gfx1013
podman exec "$tmpc" mkdir -p /wh
podman cp $work/pytorch-build-gfx1013/wheelhouse/. "$tmpc:/wh/"
podman exec "$tmpc" /bin/sh -c '
    set -eu
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip libgomp1 libopenblas0 libgfortran5
    python3 -m venv /opt/torch-venv
    printf "%s\n" /opt/bc250/rocblas-gfx1013/lib /opt/bc250/hipblas-gfx1013/lib /opt/bc250/hipblaslt-gfx1013/lib /opt/bc250/roctracer-gfx1013/lib /opt/bc250/rocfft-gfx1013/lib /opt/bc250/hipfft-gfx1013/lib /opt/bc250/rocrand-gfx1013/lib /opt/bc250/hiprand-gfx1013/lib /opt/bc250/rocsparse-gfx1013/lib /opt/bc250/hipsparse-gfx1013/lib /opt/bc250/rocsolver-gfx1013/lib /opt/bc250/hipsolver-gfx1013/lib /opt/bc250/miopen-gfx1013/lib /opt/bc250/openmp-gfx1013/lib > /etc/ld.so.conf.d/bc250-r12.conf
    ldconfig
    /opt/torch-venv/bin/pip install "numpy<2" /wh/torch-2.9.1a0+gitunknown-cp312-cp312-linux_x86_64.whl
    /opt/torch-venv/bin/python -c "import torch; print(torch.__version__)"
    rm -rf /wh
    ldconfig
    printf "%s\n" /opt/bc250/openmp-gfx1013/lib/libomp.so > /etc/ld.so.preload
    echo R12_AMD64_OK
'
podman stop -t 10 "$tmpc"
podman commit "$tmpc" "$new_img"
podman rm -f "$tmpc"
podman image inspect "$new_img" --format "{{.Id}} {{.Size}}"
echo R12_DONE
