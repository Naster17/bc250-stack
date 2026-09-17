#!/bin/sh
set -eu
# r1.3 runtime image: r1.2 + rocALUTION + RPP + hipDNN (+deps) + tuned MIOpen DB.
# GPU-free build: no devices mapped. Network allowed for apt/PyPI only.
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
work=/home/nik/bc250work
src_img=localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.2
new_img=localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.3
tmpc=bc250-r13-staging
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
podman rm -f "$tmpc" 2>/dev/null || true
podman create --name "$tmpc" "$src_img" sleep 2147483647
podman start "$tmpc"
copy_prefix() {
    dest="$2"
    podman exec "$tmpc" mkdir -p "$dest"
    rm -rf /tmp/r13copy
    mkdir -p /tmp/r13copy
    cp -a "$1/." /tmp/r13copy/
    podman cp /tmp/r13copy/. "$tmpc:$dest"
    rm -rf /tmp/r13copy
}
copy_prefix $work/rocalution-build-gfx1013/rocalution-prefix /opt/bc250/rocalution-gfx1013
copy_prefix $work/rpp-build-gfx1013/rpp-prefix /opt/bc250/rpp-gfx1013
copy_prefix $work/hipdnn-build-gfx1013/hipdnn-prefix /opt/bc250/hipdnn-gfx1013
copy_prefix $work/hipdnn-deps-prefix /opt/bc250/hipdnn-deps
podman exec "$tmpc" mkdir -p /opt/bc250/miopen-db
podman cp $work/miopen-tune-db/gfx1013_20.HIP.3_5_2_.ufdb.txt "$tmpc:/opt/bc250/miopen-db/"
podman exec "$tmpc" /bin/sh -c '
    set -eu
    printf "%s\n" /opt/bc250/rocalution-gfx1013/lib /opt/bc250/rpp-gfx1013/lib /opt/bc250/hipdnn-gfx1013/lib /opt/bc250/hipdnn-deps/lib >> /etc/ld.so.conf.d/bc250-r12.conf
    ldconfig
    ldconfig -p | grep -E "librocalution|librpp|libhipdnn_backend|libflatbuffers|libspdlog" | head -n 8
    /opt/torch-venv/bin/python -c "import torch; print(torch.__version__)"
    echo R13_AMD64_OK
'
podman stop -t 10 "$tmpc"
podman commit "$tmpc" "$new_img"
podman rm -f "$tmpc"
podman image inspect "$new_img" --format "{{.Id}} {{.Size}}"
echo R13_DONE
