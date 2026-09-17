#!/bin/sh
set -eu
# Resume dist build: fix rocm-smi include path in build.ninja, continue.
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
out=/home/nik/bc250work/pytorch-build-dist
img=localhost/bc250-rocm-build-tools:7.2.4
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm --name bc250-dist-resume2 --network none \
    --device /dev/kfd --device /dev/dri/renderD128 --security-opt seccomp=unconfined \
    -v "$out/build/pytorch:/out/build/pytorch" \
    -v /home/nik/bc250work/python-dev/stage-venv:/venv:ro \
    -v /home/nik/bc250work/python-dev/overlay/usr:/opt/pydev:ro \
    -v /usr/local/include:/rocm-smi-inc:ro \
    -v /home/nik/bc250work/rocblas-build-gfx1013/rocblas-prefix:/rocblas:ro \
    -v /home/nik/bc250work/hipblas-build-gfx1013/hipblas-prefix:/hipblas:ro \
    -v /home/nik/bc250work/hipblas-build-gfx1013/hipblas-common-prefix:/hipblas-common:ro \
    -v /home/nik/bc250work/hipblaslt-build-gfx1013/hipblaslt-prefix:/hipblaslt:ro \
    -v /home/nik/bc250work/roctracer-build-gfx1013/roctracer-prefix:/roctracer:ro \
    -v /home/nik/bc250work/rocfft-build-gfx1013/rocfft-prefix:/rocfft:ro \
    -v /home/nik/bc250work/hipfft-build-gfx1013/hipfft-prefix:/hipfft:ro \
    -v /home/nik/bc250work/rocrand-build-gfx1013/rocrand-prefix:/rocrand:ro \
    -v /home/nik/bc250work/rocrand-build-gfx1013/hiprand-prefix:/hiprand:ro \
    -v /home/nik/bc250work/rocsparse-build-gfx1013/rocsparse-prefix:/rocsparse:ro \
    -v /home/nik/bc250work/rocsparse-build-gfx1013/hipsparse-prefix:/hipsparse:ro \
    -v /home/nik/bc250work/rocsolver-build-gfx1013/rocsolver-prefix:/rocsolver:ro \
    -v /home/nik/bc250work/rocsolver-build-gfx1013/hipsolver-prefix:/hipsolver:ro \
    -v /home/nik/bc250work/rocm-headers-gfx1013/rocprim-prefix:/rocprim:ro \
    -v /home/nik/bc250work/rocm-headers-gfx1013/hipcub-prefix:/hipcub:ro \
    -v /home/nik/bc250work/rocm-headers-gfx1013/rocthrust-prefix:/rocthrust:ro \
    -v /home/nik/bc250work/miopen-build-gfx1013/miopen-prefix:/miopen:ro \
    "$img" /bin/sh -c '
        set -eu
        export PATH=/opt/rocm/bin:/usr/bin:/bin:/venv/bin
        export ROCM_PATH=/opt/rocm ROCM_HOME=/opt/rocm HIP_PATH=/opt/rocm
        export HIP_CLANG_PATH=/opt/rocm/llvm/bin
        export MIOPEN_PATH=/miopen
        export PYTORCH_ROCM_ARCH=gfx1013
        export USE_ROCM=1 USE_CUDA=0
        export USE_DISTRIBUTED=1 USE_GLOO=1 USE_MPI=0 USE_NCCL=0 USE_RCCL=0
        export CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ CMAKE_C_COMPILER=/opt/rocm/bin/amdclang
        export HIPCXX=/opt/rocm/bin/hipcc HIPCC=/opt/rocm/bin/hipcc
        export Python_EXECUTABLE=/venv/bin/python
        export MIOPEN_PATH=/miopen
        export CMAKE_PREFIX_PATH=/venv/lib/python3.12/site-packages:/opt/pydev:/venv:/miopen:/rocsolver:/hipsolver:/rocsparse:/hipsparse:/rocblas:/hipblas:/hipblas-common:/hipblaslt:/roctracer:/rocfft:/hipfft:/rocrand:/hiprand:/rocprim:/hipcub:/rocthrust:/opt/rocm
        cd /out/build/pytorch/build
        grep -c "\-I/rocm-smi" build.ninja
        sed -i "s|-I/rocm-smi |-I/rocm-smi-inc |g" build.ninja
        touch build.ninja
        grep -c "\-I/rocm-smi-inc" build.ninja
        ls /rocm-smi-inc/rocm_smi/rocm_smi.h
        cmake --build . --target install --config Release -j 16
        echo DIST_RESUME2_DONE
    '
