#!/bin/sh
set -eu
# Prebuild hipDNN third-party deps (flatbuffers, spdlog, nlohmann_json,
# tsl-robin-map) into a prefix so hipDNN find_package hits and never
# runs its network _build_local path. GPU-free, network-free.
fb=/home/nik/bc250work/third-party/flatbuffers
spdlog=/home/nik/bc250work/third-party/flatbuffers
json=/home/nik/bc250work/third-party/nlohmann
tsl=/home/nik/bc250work/third-party/tsl
out=/home/nik/bc250work/hipdnn-deps-prefix
img=localhost/bc250-rocm-build-tools:7.2.4
mkdir -p "$out"
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm --name bc250-hipdnn-deps --network none \
    -v "$fb:/fb:ro" -v "$json:/json:ro" -v "$tsl:/tsl:ro" \
    -v "$out:/out" \
    "$img" /bin/sh -c '
        set -eu
        export CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++
        rm -rf /tmp/d && mkdir -p /tmp/d
        cd /tmp/d && tar -xzf /fb/flatbuffers-25.9.23.tar.gz && tar -xzf /fb/spdlog-1.15.3.tar.gz
        mkdir -p /tmp/d/json && tar -xzf /json/json-3.12.0.tar.gz -C /tmp/d/json --strip-components=1
        mkdir -p /tmp/d/tsl && tar -xzf /tsl/robin-map-1.4.1.tar.gz -C /tmp/d/tsl --strip-components=1
        cmake -S /tmp/d/json -B /tmp/d/json-b -G Ninja \
            -DCMAKE_INSTALL_PREFIX=/out -DJSON_Install=ON \
            -DJSON_BuildTests=OFF &&
        cmake --build /tmp/d/json-b && cmake --install /tmp/d/json-b
        cmake -S /tmp/d/tsl -B /tmp/d/tsl-b -G Ninja \
            -DCMAKE_INSTALL_PREFIX=/out &&
        cmake --build /tmp/d/tsl-b && cmake --install /tmp/d/tsl-b
        cmake -S /tmp/d/spdlog-1.15.3 -B /tmp/d/spdlog-b -G Ninja \
            -DCMAKE_INSTALL_PREFIX=/out -DSPDLOG_INSTALL=ON \
            -DSPDLOG_BUILD_TESTS=OFF -DSPDLOG_BUILD_EXAMPLE=OFF &&
        cmake --build /tmp/d/spdlog-b && cmake --install /tmp/d/spdlog-b
        cmake -S /tmp/d/flatbuffers-25.9.23 -B /tmp/d/fb-b -G Ninja \
            -DCMAKE_INSTALL_PREFIX=/out \
            -DFLATBUFFERS_BUILD_FLATC=ON -DFLATBUFFERS_INSTALL=ON \
            -DFLATBUFFERS_BUILD_FLATLIB=ON -DFLATBUFFERS_BUILD_TESTS=OFF \
            -DFLATBUFFERS_BUILD_FLATHASH=OFF -DFLATBUFFERS_ENABLE_PCH=OFF &&
        cmake --build /tmp/d/fb-b && cmake --install /tmp/d/fb-b
        ls /out/lib/cmake/flatbuffers/ /out/lib/cmake/spdlog/ /out/share/cmake/nlohmann/ 2>&1 | head -n 12
        ls /out/bin/flatc && echo HIPDNN_DEPS_OK
    '
