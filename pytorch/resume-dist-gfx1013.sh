#!/bin/sh
set -eu
# Resume dist build: reconfigure with rocm-smi include, continue ninja.
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
out=/home/nik/bc250work/pytorch-build-dist
img=localhost/bc250-rocm-build-tools:7.2.4
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
exec podman run --rm --name bc250-torch-dist-resume --network none \
    -v "$out/build/pytorch:/out/build/pytorch" \
    -v /home/nik/bc250work/python-dev/stage-venv:/venv:ro \
    -v /home/nik/bc250work/python-dev/overlay/usr:/opt/pydev:ro \
    -v /usr/local/include/rocm_smi:/rocm-smi:ro \
    "$img" /bin/sh -c '
        set -eu
        export PATH=/opt/rocm/bin:/usr/bin:/bin:/venv/bin
        export ROCM_PATH=/opt/rocm ROCM_HOME=/opt/rocm HIP_PATH=/opt/rocm
        cd /out/build/pytorch/build
        cmake . -DCMAKE_CXX_FLAGS="-I/opt/pydev/include/python3.12 -I/venv/lib/python3.12/site-packages/numpy/core/include -I/rocm-smi" > /out/resume-cfg.log 2>&1
        grep -E "USE_DISTRIBUTED|USE_GLOO" CMakeCache.txt | head -n 3
        cmake --build . --target install --config Release -j 16
        echo DIST_RESUME_DONE
    '
