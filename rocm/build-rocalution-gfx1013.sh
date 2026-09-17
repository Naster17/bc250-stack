#!/bin/sh
set -eu
# rocALUTION native gfx1013 (GPU-free, network-free).
src=/home/nik/bc250work/rocm-libraries-gfx1013
rcmake=/home/nik/bc250work/rocm-cmake-7.2.4
rocblas=/home/nik/bc250work/rocblas-build-gfx1013/rocblas-prefix
rocprim=/home/nik/bc250work/rocm-headers-gfx1013/rocprim-prefix
rocsparse=/home/nik/bc250work/rocsparse-build-gfx1013/rocsparse-prefix
rocrand=/home/nik/bc250work/rocrand-build-gfx1013/rocrand-prefix
out=/home/nik/bc250work/rocalution-build-gfx1013
img=localhost/bc250-rocm-build-tools:7.2.4
mkdir -p "$out"
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm --name bc250-rocalution-build --network none \
    -v "$src:/src:ro" -v "$rcmake:/rocm-cmake:ro" \
    -v "$rocblas:/rocblas:ro" -v "$rocprim:/rocprim:ro" \
    -v "$rocsparse:/rocsparse:ro" -v "$rocrand:/rocrand:ro" \
    -v "$out:/out" \
    "$img" /bin/sh -c '
        set -eu
        export CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++
        rm -rf /out/rocalution-build
        cmake -S /src/projects/rocalution -B /out/rocalution-build -G Ninja \
            -DCMAKE_INSTALL_PREFIX=/out/rocalution-prefix \
            -DCMAKE_BUILD_TYPE=Release \
            -DROCmCMakeBuildTools_DIR=/rocm-cmake/share/rocmcmakebuildtools/cmake \
            -DROCM_DIR=/opt/rocm \
            -DHIP_DIR=/opt/rocm/lib/cmake/hip \
            -DCMAKE_PREFIX_PATH="/opt/rocm;/rocprim;/rocblas;/rocsparse;/rocrand" \
            -Drocblas_DIR=/rocblas/lib/cmake/rocblas \
            -Drocsparse_DIR=/rocsparse/lib/cmake/rocsparse \
            -Drocprim_DIR=/rocprim/lib/cmake/rocprim \
            -Drocrand_DIR=/rocrand/lib/cmake/rocrand \
            -DAMDGPU_TARGETS=gfx1013:xnack- \
            -DGPU_TARGETS=gfx1013:xnack- \
            -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF \
            -DBUILD_CLIENTS_SAMPLES=OFF &&
        cmake --build /out/rocalution-build && cmake --install /out/rocalution-build &&
        ls /out/rocalution-prefix/lib/librocalution.so &&
        echo ROCALUTION_INSTALL_OK
    '
