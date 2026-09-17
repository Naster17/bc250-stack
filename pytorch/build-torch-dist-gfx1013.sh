#!/bin/sh
set -eu
# Phase 5b: PyTorch 2.9.1 + Gloo distributed for 2-board training.
# Source = staged already-patched tree (no re-patch); separate out dir.
# GPU-free, network-free build.
src=/home/nik/bc250work/pytorch-build-gfx1013/build/pytorch
venv=/home/nik/bc250work/python-dev/stage-venv
out=/home/nik/bc250work/pytorch-build-dist
img=localhost/bc250-rocm-build-tools:7.2.4
mkdir -p "$out"
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm --name bc250-torch-dist --network none \
    -v "$src:/src:ro" -v "$venv:/venv:ro" -v "$out:/out" \
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
    -v /home/nik/bc250work/python-dev/overlay/usr:/opt/pydev:ro \
    -v /home/nik/bc250work/third-party/eigen-3.4.0:/eigen:ro \
    -v /usr/local/include:/rocm-smi-inc:ro \
    "$img" /bin/sh -c '
        set -eu
        export VIRTUAL_ENV=/venv
        export PATH=/opt/rocm/bin:/usr/bin:/bin:/venv/bin
        export ROCM_PATH=/opt/rocm ROCM_HOME=/opt/rocm HIP_PATH=/opt/rocm
        export HIP_CLANG_PATH=/opt/rocm/llvm/bin
        export MIOPEN_PATH=/miopen
        export CMAKE_PREFIX_PATH=/venv/lib/python3.12/site-packages:/opt/pydev:/venv:/miopen:/rocsolver:/hipsolver:/rocsparse:/hipsparse:/rocblas:/hipblas:/hipblas-common:/hipblaslt:/roctracer:/rocfft:/hipfft:/rocrand:/hiprand:/rocprim:/hipcub:/rocthrust:/opt/rocm
        export USE_ROCM=1 USE_CUDA=0
        export PYTORCH_ROCM_ARCH=gfx1013
        export USE_DISTRIBUTED=1 USE_NCCL=0 USE_RCCL=0 USE_GLOO=1 USE_MPI=0
        export USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0
        export BUILD_TEST=0 BUILD_CAFFE2=0 USE_KINETO=0
        export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0
        export USE_PYTORCH_QNNPACK=0
        export USE_ROCM_CK_GEMM=0 USE_ROCM_CK_SDPA=0
        export USE_NUMPY=1 USE_SYSTEM_EIGEN_INSTALL=0 USE_SYSTEM_LIBS=0 BLAS=Eigen
        export USE_NUMA=0 USE_ITT=0
        export INTERN_DISABLE_ONNX=1
        export MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16
        export CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ CMAKE_C_COMPILER=/opt/rocm/bin/amdclang
        export HIPCXX=/opt/rocm/bin/hipcc HIPCC=/opt/rocm/bin/hipcc
        export CXXFLAGS="-I/rocm-smi-inc"
        export Python_EXECUTABLE=/venv/bin/python
        export Python_INCLUDE_DIR=/opt/pydev/include/python3.12
        export Python_LIBRARY=/opt/pydev/lib/x86_64-linux-gnu/libpython3.12.a
        export Python_NumPy_INCLUDE_DIR=/venv/lib/python3.12/site-packages/numpy/core/include
        export Python_NumPy_INCLUDE_DIRS=/venv/lib/python3.12/site-packages/numpy/core/include
        export Python_ROOT_DIR=/opt/pydev
        export Python_FIND_VIRTUALENV=FIRST
        rm -rf /out/build /out/wheelhouse /out/build.log
        mkdir -p /out/build/pytorch /out/wheelhouse
        cp -a /src/. /out/build/pytorch/
        rm -rf /out/build/pytorch/build
        chown -R root:root /out/build /out/wheelhouse
        cd /out/build/pytorch
        echo STAGE_COPY_DONE
        /venv/bin/python tools/amd_build/build_amd.py > /out/build.log 2>&1
        echo HIPIFY_DONE >> /out/build.log
        export CFLAGS="-I/opt/pydev/include/python3.12 -I/venv/lib/python3.12/site-packages/numpy/core/include"
        export CPPFLAGS="-I/opt/pydev/include/python3.12 -I/venv/lib/python3.12/site-packages/numpy/core/include"
        /venv/bin/python setup.py bdist_wheel --dist-dir /out/wheelhouse >> /out/build.log 2>&1
        echo BUILD_DONE >> /out/build.log
        ls -la /out/wheelhouse/
        echo TORCH_DIST_DONE
    '
