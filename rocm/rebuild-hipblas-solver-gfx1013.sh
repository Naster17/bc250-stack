#!/bin/sh
set -eu
# Rebuild native gfx1013 hipBLAS wrapper with solver API (BUILD_WITH_SOLVER=ON).
# Same isolation recipe as HIPBLAS-BUILD-2026-09-13.md, plus native rocsolver prefix.
src=/home/nik/bc250work/rocm-libraries-gfx1013
rcmake=/home/nik/bc250work/rocblas-build-gfx1013/rocm-cmake-prefix
rocblas=/home/nik/bc250work/rocblas-build-gfx1013/rocblas-prefix
rocsolver=/home/nik/bc250work/rocsolver-build-gfx1013/rocsolver-prefix
hipcommon=/home/nik/bc250work/hipblas-build-gfx1013/hipblas-common-prefix
out=/home/nik/bc250work/hipblas-build-gfx1013
img=localhost/bc250-rocm-build-tools:7.2.4
mkdir -p "$out"
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm --name bc250-hipblas-solver --network none \
    -v "$src:/src:ro" -v "$rcmake:/rocm-cmake:ro" \
    -v "$rocblas:/rocblas:ro" -v "$rocsolver:/rocsolver:ro" \
    -v "$hipcommon:/hipblas-common:ro" -v "$out:/out" \
    "$img" /bin/sh -c '
        set -eu
        export CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++
        rm -rf /out/hipblas-solver-build
        cmake -S /src/projects/hipblas -B /out/hipblas-solver-build -G Ninja \
            -DCMAKE_INSTALL_PREFIX=/out/hipblas-prefix \
            -DCMAKE_BUILD_TYPE=Release \
            -DROCmCMakeBuildTools_DIR=/rocm-cmake/share/rocmcmakebuildtools/cmake \
            -DROCM_DIR=/opt/rocm \
            -DHIP_DIR=/opt/rocm/lib/cmake/hip \
            -DCMAKE_PREFIX_PATH="/opt/rocm;/hipblas-common;/rocblas;/rocsolver" \
            -Dhipblas-common_DIR=/hipblas-common/lib/cmake/hipblas-common \
            -DCUSTOM_ROCBLAS=/rocblas/lib/cmake/rocblas \
            -DCUSTOM_ROCSOLVER=/rocsolver/lib/cmake/rocsolver \
            -DAMDGPU_TARGETS=gfx1013:xnack- \
            -DBUILD_WITH_SOLVER=ON \
            -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF \
            -DBUILD_WITH_SPARSE=ON \
            -DBUILD_FORTRAN_BINDINGS=OFF &&
        cmake --build /out/hipblas-solver-build && cmake --install /out/hipblas-solver-build &&
        ls /out/hipblas-prefix/lib/libhipblas.so &&
        echo HIPBLAS_SOLVER_OK
    '
