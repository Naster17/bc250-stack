# bc250-stack

BC-250 (`1002:13fe`, `gfx1013:xnack-`) integration repo: kernel patch set,
native gfx1013 ROCm userspace build scripts, r1/r1.2 image recipes, PyTorch
build, validation docs. This is experimentally validated local work, not AMD
official support.

Upstream code lives in sibling forks; this repo holds only our layer:

- `Naster17/bc250-linux` (fork of `torvalds/linux`) - branch `bc250-r2` (7.1.5, default; `bc250-r1` = 6.18.50 legacy)
- `Naster17/bc250-rocm` (fork of `boondocklabs/rocm-libraries`) - branch `bc250-gfx1013`
- `Naster17/bc250-torch` (fork of `pytorch/pytorch` at `v2.9.1`) - branch `bc250-gfx1013`
- `Naster17/llama.cpp` - branch `gfx1013-rdna1`

## Layout

- `kernel/` - 40-CU unlock + PASID/runlist TLB + TTM guard patches, GRUB
  one-shot helpers, initramfs notes, stock-safety rules.
- `rocm/` - native prefix build scripts (rocBLAS/hipBLAS/rocFFT/hipFFT/
  rocRAND/rocSPARSE/rocSOLVER/hipblaslt/roctracer/MIOpen/headers), prefix
  manifest with sizes and hashes.
- `pytorch/` - `0004`-`0011` patch set, `build-torch-gfx1013.sh`,
  `build-wheel-gfx1013.sh`, overlay/python notes, wheel hash.
- `images/` - r1.2 image recipe (`build-r12-image.sh`), `run-torch.sh`,
  `run-llama.sh` launchers, `ld.so` + libomp notes.
- `docs/` - build reports (`PYTORCH-BUILD-2026-09-16.md`, phase docs),
  `SUPPORTED-PROFILE.md`, validation numbers (GEMM, 50-step train,
  llama gate, PPL reference).
- `profile/` - `bc250-rocm-profile` repo snapshot pointer (config.env,
  manifests); the profile repo itself stays the deployable unit.

## Build order (GPU-free except validation)

1. Kernel r1 (`kernel/`), one-shot boot.
2. Native prefixes (`rocm/`, scripts run on the board, network-free
   except pinned upstream debs for hipblaslt/roctracer headers+libs).
3. hipBLAS solver rebuild, MIOpen, headers.
4. PyTorch wheel (`pytorch/`), ~2h on 16 cores.
5. r1.2 image (`images/`), GPU gates.

## Validated results (2026-09-16, r1 + r1.2)

- torch 2.9.1a0, `cuda` True, 1 device, arch `['gfx1013']`.
- 64x64 GEMM OK; 50-step MLP train bit-identical to CPU
  (`1.029231 / 1.004733 / 0.985372`).
- Llama gate re-passed on r1.2.
- PPL reference `63.9747 +/- 4.69923` (wikitext-2, ctx 2048).

## Limits (honest)

- hipblaslt: headers + runtime lib only, no gfx1013 kernel rebuild.
- Torch needs `numpy<2`, LLVM `libomp.so` preloaded, `HSA_ENABLE_SDMA=1`.
- No distributed/NCCL, flash-attn, CK paths.
- r1 one-shot only; stock kernel stays default; user power-cycles on wedge.

## 8-core CPU unlock

Real and separate from the GPU work: SMU mask `0x0115A870`
(`0x77` stock, `0xFF` unlocked) via `GabriWar/bc250-core-cu-unlock`
(`bc250-8core-unlock.sh`), warm reboot to enumerate, ACPI update for
CPUs 12-15. Our board mask already reads UNLOCKED, enumeration pending.
Details + telemetry patch copy in `bc250-linux` (`docs/8CORE-UNLOCK.md`,
`kernel-patches/0005`). Full doc:
https://elektricm.github.io/amd-bc250-docs/system/8core-unlock/
