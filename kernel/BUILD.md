# Kernel r1 build

Base: Alpine `linux-lts` 6.18.50 source (`alpine-rocm/sources/linux-6.18`).
Work tree `linux-6.18-bc250-test` carries (source changes only):

1. `0000-bc250-40cu-amdgpu.patch` (duggasco 40-CU unlock, PCI `0x13FE`
   gated, `amdgpu.bc250_cc_write_mode=3`).
2. `0001-amdgpu-bc250-pasid-flush-param.patch`
   (`amdgpu.bc250_flush_pasid_kiq=0`).
3. `0002-amdgpu-ttm-skip-null-pages.patch` (amdgpu#222 guard).
4. KFD runlist flush `bc250_flush_by_runlist=3` (akandr runlist-rebuild
   approach; lives in `kfd_device_queue_manager.c` + call sites in
   `kfd_chardev.c`, `kfd_svm.c`).

Boot: versioned `/boot/vmlinuz-bc250-rocm-r1` + initramfs (navi12 SDMA
ucode inside r1 initramfs only) + `/lib/modules/6.18.50-bc250rocm-test`,
one-shot GRUB `bc250-rocm-r1`, stock stays default. Params:
`3/0/3/0/0`, `gpu_recovery=0`, no `sched_policy=2`.
