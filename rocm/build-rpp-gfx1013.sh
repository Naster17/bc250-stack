#!/bin/sh
set -eu
# RPP native gfx1013, HIP backend (GPU-free, network-free).
src=/home/nik/bc250work/rocm-libraries-gfx1013
rcmake=/home/nik/bc250work/rocm-cmake-7.2.4
half=/home/nik/bc250work/third-party/half/include
out=/home/nik/bc250work/rpp-build-gfx1013
img=localhost/bc250-rocm-build-tools:7.2.4
mkdir -p "$out"
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm --name bc250-rpp-build --network none \
    -v "$src:/src:ro" -v "$rcmake:/rocm-cmake:ro" \
    -v "$half:/half:ro" -v "$out:/out" \
    "$img" /bin/sh -c '
        set -eu
        export CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++
        rm -rf /out/rpp-build
        cmake -S /src/projects/rpp -B /out/rpp-build -G Ninja \
            -DCMAKE_INSTALL_PREFIX=/out/rpp-prefix \
            -DCMAKE_BUILD_TYPE=Release \
            -DROCmCMakeBuildTools_DIR=/rocm-cmake/share/rocmcmakebuildtools/cmake \
            -DROCM_DIR=/opt/rocm \
            -DHIP_DIR=/opt/rocm/lib/cmake/hip \
            -DCMAKE_PREFIX_PATH="/opt/rocm;/rocm-cmake" \
            -DBACKEND=HIP \
            -DHALF_DIR=/half \
            -DRPP_AUDIO_SUPPORT=OFF \
            -DGPU_TARGETS=gfx1013:xnack- \
            -DBUILD_TESTS=OFF &&
        cmake --build /out/rpp-build && cmake --install /out/rpp-build &&
        ls /out/rpp-prefix/lib/librpp.so &&
        echo RPP_INSTALL_OK
    '
