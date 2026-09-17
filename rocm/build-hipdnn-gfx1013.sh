#!/bin/sh
set -eu
# hipDNN SDK (frontend + backend core, no engines) native gfx1013.
# GPU-free, network-free. Engines arrive via plugins later.
src=/home/nik/bc250work/rocm-libraries-gfx1013
rcmake=/home/nik/bc250work/rocm-cmake-7.2.4
out=/home/nik/bc250work/hipdnn-build-gfx1013
deps=/home/nik/bc250work/hipdnn-deps-prefix
img=localhost/bc250-rocm-build-tools:7.2.4
mkdir -p "$out"
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm --name bc250-hipdnn-build --network none \
    -v "$src:/src:ro" -v "$rcmake:/rocm-cmake:ro" \
    -v "$deps:/deps:ro" -v "$out:/out" \
    "$img" /bin/sh -c '
        set -eu
        export CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++
        rm -rf /out/hipdnn-build
        cmake -S /src/projects/hipdnn -B /out/hipdnn-build -G Ninja \
            -DCMAKE_INSTALL_PREFIX=/out/hipdnn-prefix \
            -DCMAKE_BUILD_TYPE=Release \
            -DROCmCMakeBuildTools_DIR=/rocm-cmake/share/rocmcmakebuildtools/cmake \
            -DROCM_DIR=/opt/rocm \
            -DHIP_DIR=/opt/rocm/lib/cmake/hip \
            -DCMAKE_PREFIX_PATH="/opt/rocm;/rocm-cmake;/deps" \
            -DHIPDNN_BUILD_BACKEND=ON \
            -DHIPDNN_BUILD_FRONTEND=ON \
            -DHIPDNN_SKIP_TESTS=ON \
            -DHIPDNN_BUILD_PYTHON_BINDINGS=OFF \
            -DHIPDNN_ENABLE_SDPA=OFF \
            -DHIPDNN_GENERATE_SDK_HEADERS=OFF \
            -DENABLE_CLANG_FORMAT=OFF \
            -DENABLE_CLANG_TIDY=OFF &&
        cmake --build /out/hipdnn-build && cmake --install /out/hipdnn-build &&
        ls /out/hipdnn-prefix/lib/libhipdnn*.so &&
        echo HIPDNN_INSTALL_OK
    '
