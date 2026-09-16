# bc250-stack

Integration repo for the BC-250 (`1002:13fe`, `gfx1013:xnack-`) ROCm
bring-up: kernel patches, native gfx1013 userspace builds, runtime image
recipes, PyTorch build, validation docs.

Experimentally validated local work — **not AMD official support**.

## Repos

Upstream code lives in sibling forks; this repo holds only our layer
(patches, scripts, docs):

| Repo | Upstream | Branch | Contains |
|---|---|---|---|
| [`bc250-linux`](https://github.com/Naster17/bc250-linux) | `torvalds/linux` | `bc250-r2` (7.1.5, current default; `bc250-r1` = 6.18.50 legacy) | Kernel patch series 0000–0005 |
| [`bc250-rocm`](https://github.com/Naster17/bc250-rocm) | `boondocklabs/rocm-libraries` | `bc250-gfx1013` | MIOpen gfx1013 solver gates |
| [`bc250-torch`](https://github.com/Naster17/bc250-torch) | `pytorch/pytorch` at `v2.9.1` | `bc250-gfx1013` | Prefix-aware build fixes |
| [`llama.cpp`](https://github.com/Naster17/llama.cpp) | `ggml-org/llama.cpp` | `gfx1013-rdna1` | RDNA1 tile-dispatch fix |

## Layout

```text
kernel/    40-CU unlock + PASID/runlist TLB + TTM guard patches,
           GRUB one-shot helpers, initramfs notes, stock-safety rules
rocm/      native prefix build scripts (rocBLAS/hipBLAS/rocFFT/hipFFT,
           rocRAND/rocSPARSE/rocSOLVER/hipblaslt/roctracer/MIOpen/headers),
           MIOpen + llama diffs, prefix manifest with sizes and hashes
pytorch/   0006-0012 patch set, build-torch-gfx1013.sh,
           build-wheel-gfx1013.sh, overlay/python notes, wheel hash
images/    r1.2 image recipe (build-r12-image.sh), run-torch.sh,
           run-llama.sh launchers, ld.so + libomp notes
docs/      build reports (PYTORCH-BUILD-2026-09-16.md, phase docs),
           SUPPORTED-PROFILE.md, validation numbers
profile/   deployable profile snapshot pointer (config.env, manifests)
```

## Build order (GPU-free except validation)

1. **Kernel** (`kernel/`, or `bc250-linux#bc250-r2`) — one-shot boot.
2. **Native prefixes** (`rocm/`) — scripts run on the board, network-free
   except pinned upstream debs for hipblaslt/roctracer headers+libs.
3. **hipBLAS solver rebuild** + MIOpen + headers.
4. **PyTorch wheel** (`pytorch/`) — ~2h on 16 cores.
5. **r1.2 image** (`images/`) — GPU gates.

## Validated results (2026-09-16, r2 + r1.2)

- torch 2.9.1a0: `cuda` True, 1 device, arch `['gfx1013']`.
- 64x64 GEMM OK; 50-step MLP train bit-identical to CPU
  (`1.029231 / 1.004733 / 0.985372`).
- Llama gate re-passed on r1.2 (95.8 t/s prompt, 40.6 t/s gen).
- SDMA 4 KiB–16 MiB sweep 6/6 OK with `HSA_ENABLE_SDMA=1`.
- PPL reference `63.9747 +/- 4.69923` (wikitext-2, ctx 2048, from r1).

## Limits (honest)

- hipblaslt: headers + runtime lib only, no gfx1013 kernel rebuild.
- Torch needs `numpy<2`, LLVM `libomp.so` preloaded, `HSA_ENABLE_SDMA=1`.
- No distributed/NCCL, flash-attn, or CK paths — single-GPU train/infer
  only (standard SDPA falls back to the math backend).
- r2 is the default boot since 2026-09-16 (after full gates); stock
  entries stay intact. A wedged GPU = power cycle.
