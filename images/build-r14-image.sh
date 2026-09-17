#!/bin/sh
set -eu
# r1.4 runtime image: r1.3 + rocm-smi lib + distributed torch venv.
# Keeps validated /opt/torch-venv; adds /opt/torch-venv-dist (Gloo wheel).
# GPU-free build: no devices mapped. Network allowed for apt/PyPI only.
test "$(id -u)" -eq 0 || {
    echo "run through doas" >&2
    exit 1
}
work=/home/nik/bc250work
src_img=localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.3
new_img=localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.4
tmpc=bc250-r14-staging
mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup
podman rm -f "$tmpc" 2>/dev/null || true
podman create --name "$tmpc" "$src_img" sleep 2147483647
podman start "$tmpc"
rm -rf /tmp/r14copy
mkdir -p /tmp/r14copy
cp -a $work/rocm-smi-prefix/. /tmp/r14copy/
podman exec "$tmpc" mkdir -p /opt/bc250/rocm-smi-gfx1013
podman cp /tmp/r14copy/. "$tmpc:/opt/bc250/rocm-smi-gfx1013"
rm -rf /tmp/r14copy
podman exec "$tmpc" mkdir -p /wh-dist
podman cp $work/pytorch-build-dist/wheelhouse/. "$tmpc:/wh-dist/"
podman exec "$tmpc" /bin/sh -c '
    set -eu
    printf "%s\n" /opt/bc250/rocm-smi-gfx1013/lib >> /etc/ld.so.conf.d/bc250-r12.conf
    ldconfig
    python3 -m venv /opt/torch-venv-dist
    /opt/torch-venv-dist/bin/pip install "numpy<2" /wh-dist/torch-2.9.1a0+gitunknown-cp312-cp312-linux_x86_64.whl
    rm -rf /wh-dist
    printf "%s\n" /opt/bc250/rocm-smi-gfx1013/lib/librocm_smi64.so.1 >> /etc/ld.so.preload
    cat /etc/ld.so.preload
    /opt/torch-venv-dist/bin/python -c "import torch.distributed as d; print(d.is_available(), d.is_gloo_available())"
    echo R14_AMD64_OK
'
podman stop -t 10 "$tmpc"
podman commit "$tmpc" "$new_img"
podman rm -f "$tmpc"
podman image inspect "$new_img" --format "{{.Id}} {{.Size}}"
echo R14_DONE
