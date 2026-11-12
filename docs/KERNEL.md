# Kernel (`bc250-linux`)

The BC-250 kernel carries the amdgpu/amdkfd compute quirks that make gfx1013 a
usable compute device. This document records its provenance and patch series.

## Provenance (read this first)

Unlike the other component forks, the kernel is **not a clean upstream fork**:

- The **r2** source is an **already-patched snapshot** imported from the legacy
  build (Alpine Linux **7.1** base, kernel `7.1.5-bc250rocm-r2`). It is *not* a
  pristine upstream download, so there is no clean upstream tag to rebase onto.
- The **r3** work is a **6-patch delta** on top of r2
  (`7.2.10-bc250rocm-r3`).

The repository therefore preserves the **patched source tree**, the **r3 patch
series**, the **`.config`**, and **anchor hashes** so the exact running kernel is
reproducible. The import is a single snapshot commit; the r3 series is kept as
reviewable patches in [`../patches/kernel-r3/`](../patches/kernel-r3/).

## Verified source anchors (r2)

| File | SHA-256 |
| --- | --- |
| `.config` | `2ba383b4c6e519ca4cd441e7e19432f68d40551a4793ba926e3ab1f06a6f1d0c` |
| `Makefile` | `fb9e8ab7d6cf8a948f599e1adb4ac9cfcdf9fb3b5d73dcbec29b19abccea98b9` |
| `drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c` | `fe44a207d763e99b0e4abe574e2a2e0d813c089d4c2e0210ea482a027bcd6898` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_vm.c` | `263c5d83b1f1fd6ba575ee9bfc8e098ca9783ad775dd86d14a85778fa532e72a` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_ttm.c` | `389cb7ffcd70255d16f9dd941cf61d01c3cb447f85a968a9ab6fa5174fd9b444` |

The kernel identity matches the active r2 kernel on both boards. Compiled objects
(`.o/.a/.ko/.cmd/...`), `vmlinux*`, `bzImage`, `System.map`, the signing private
key, and the generated initramfs archive are **excluded** from the source import.

## r3 patch series (6)

1. `drm/amdgpu`: retain explicit BC-250 quirks and restore display behavior.
2. `drm/amdkfd`: retain BC-250 runlist workaround with calibrated timing.
3. `drm/amdgpu`: invalidate BC-250 compute VMM translations.
4. `drm/amdkfd`: retain BC-250 queue eviction on SVM CPU unmap.
5. `drm/amdkfd`: defer BC-250 SVM runlist invalidation until safe.
6. `drm/amd/display`: clear only implemented HPD IRQ sources.

## Configuration

`configs/r2.config` matches the source `.config` (not extracted from
`/proc/config.gz`). The r3 handoff carries `kernel.config` and `r2-seed.config`.

## Caveat

Board-support repositories and runtime unlocks are retained separately; the
source import is **not** proof that all optional unlock patches are applied.