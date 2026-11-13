# BC-250 stack architecture

This document explains how the port is layered, what the gfx1013 hardware
constraints are, and what each component changes. It is the starting point for
studying or contributing to the stack. The repo map with exact pins is
[`REPOS.md`](REPOS.md); the patch series are in [`../patches/`](../patches/).

## The hardware problem

The BC-250 is an ASRock APU built around **Cyan Skillfish / gfx1013**, a consumer
**RDNA1** GPU. Upstream ROCm and PyTorch target CDNA (MI-series) and newer RDNA
parts, so gfx1013 falls through several assumptions:

- **No MFMA/WMMA matrix instructions.** RDNA1 has no dedicated matrix cores;
  low-precision GEMM/dot paths that assume MFMA must be lowered through scalar or
  DPP paths instead.
- **SDMA disabled.** Async copy engines are not used; transfers go through the
  compute path.
- **`xnack` disabled** in the qualified configuration (no page-retry).
- **Compute queue / VM quirks.** The kernel's amdgpu/amdkfd compute queues, VM
  translation, and SVM runlist handling need BC-250-specific workarounds.
- **Shared CPU/GPU memory.** The APU has a unified memory pool; capacity and
  bandwidth are shared, which shapes the build envelope and training batch sizes.

The port's job is to make each layer explicitly correct for these constraints
rather than silently falling back or faulting.

## Layer 1 — Kernel (`bc250-linux`)

The amdgpu/amdkfd driver carries the BC-250 compute quirks. The r3 patch series
(6 patches) covers:

- retaining explicit BC-250 quirks and restoring correct display behavior;
- the BC-250 runlist workaround with calibrated timing;
- invalidating BC-250 compute VMM translations;
- retaining queue eviction on SVM CPU unmap;
- deferring SVM runlist invalidation until it is safe;
- clearing only implemented HPD IRQ sources in amd/display.

**Provenance caveat:** the r2 source is an already-patched snapshot (Alpine Linux
7.1 base), not a pristine upstream checkout; r3 is a delta on top. The repo keeps
the source tree, the `.config`, the r3 series, and anchor hashes so the exact
running kernel is reproducible. See [`KERNEL.md`](KERNEL.md).

## Layer 2 — ROCm systems (`bc250-rocm-systems`)

HIP, ROCr, RCCL, and the profiler. The 8-patch series fixes gfx1013-specific
behavior, including the terminating `-1` visibility selector in profiler agent
filtering and RCCL DDA IPC diagnostics. These make device enumeration, profiling,
and collective communication correct on the board.

## Layer 3 — ROCm libraries (`bc250-rocm-libraries`)

rocBLAS, hipBLASLt, MIOpen, hipSPARSELt, and friends. The 38-patch series is the
largest and includes the headline repairs:

- **MIOpen gfx1013 FP16 direct-forward with FP32 accumulation** (the PLAN-02
  repair), plus a versioned binary cache keyed after the repair.
- The explicit opt-in **FP32/FP16 CK DL forward plugin** facade and its
  descriptor/runtime contracts (B07.12/B07.13; never the global default).
- hipBLASLt native gfx1013 backend selection.
- DPP-only subwave reducer guards for strict builds.

These give correct, cached convolution and GEMM on a part with no matrix cores.

## Layer 4 — Triton (`bc250-triton`)

The JIT compiler for custom kernels. The 4-patch series:

- lowers gfx1013 low-precision dots through the scalar path (no MFMA);
- distinguishes widened gfx1013 dot accumulation and signed limits (tests);
- uses scalar BF16 elementwise multiplication on gfx1013;
- initializes RDNA1 buffer descriptor flags correctly.

This is what makes the native Triton training ops (gated-delta, conv) compile and
run correctly on RDNA1.

## Layer 5 — PyTorch (`bc250-torch`)

The framework, based on **`ROCm/pytorch`**. The complete 22-patch series
(`patches/bc250-pytorch-complete`, single base `c779020750`) adds the native
gfx1013 training surface:

- HIP channels-last average-padding fix;
- native gfx1013 **hipBLASLt** backend qualification and **hipSPARSELt** integration;
- **opt-in training ops** (explicit, never silent fallback);
- native **sparse autograd**;
- native **gated-delta** preparation, precision handling, and head-256 attention;
- native **Kineto SDK** profiling and ROCm profiler CPU-first initialization;
- ROCm sparse triangular pivot NaN fix;
- distributed/quantization hardening: BF16 NCCL PreMulSum storage, FP8 reduction
  admission checks, weight-only INT8 layout normalization, TCPStore owner
  lifetime through peer agent exit.

Attention runs through math/eager SDPA by default, with an opt-in tiled/hybrid
path; quantization is a frozen **W8** route (no NF4/packed-INT4/full-QLoRA). The
design rule throughout: **explicit opt-in, refuse rather than silently fall
back.**

## Layer 6 — Consumers

- **torchaudio** (`bc250-torchaudio`, 4 patches): RNNT counters with checked async
  memsets on the consuming stream; IIR kernel launch kept outside the dispatch
  macro during hipify.
- **torchcodec** (`bc250-torchcodec`, 1 patch): exact-timestamp single-frame seeks
  across packet-duration gaps.
- **torchvision**: built **pristine** at `v0.28.0` for gfx1013 — no source patches,
  so it is a recipe-fetched pin, not a fork.

## Layer 7 — Graph compilers

- **MIGraphX** (`bc250-migraphx`, 5 patches on `rocm-10.1`): DPP-only subwave
  reducer guards for strict C++17, rocBLAS default queries without hipBLASLt,
  stable C-API CMake target, and C++20 consumer return-type qualification.
- **ONNX Runtime** (`bc250-onnxruntime`, 1 patch on `v1.31.0`): explicit MIGraphX
  provider conversions and lifetime declarations.

## Build and release model

- **TheRock** (`bc250-therock`) is the ROCm superbuild that orchestrates the
  libraries/systems/compiler. TheRock's native model is "commits on a submodule
  branch ↔ patch series" (`fetch_sources.py` applies `patches/<tag>/<project>/`,
  `save_patches.sh` regenerates them) — which is exactly how this port is
  published.
- **Reconstruction proof (PLAN-07, D47–D56):** the whole stack was rebuilt offline
  from preserved, reviewed inputs alone (pins, patches, toolchain, external
  payloads), proving the release is reproducible without the original in-place
  build environment.
- **Release (PLAN-08, D57–D61):** a self-contained, immutable, versioned prefix
  (`10.2-gfx1013-20261111-138a66f52`), qualified by independent-path activation on
  both boards with live rollback, plus SBOM, license bundle, and source recipes.

## Design principles

1. **Explicit over silent.** Native ops are opt-in and refuse unsupported
   shapes/dtypes rather than falling back quietly.
2. **Bounded evidence.** Every claim (correctness, compatibility, performance) is
   backed by a bounded, owned fixture and a receipt — not architecture-name
   promises.
3. **Rebasable.** Each fork branch sits on a recorded upstream pin so it can be
   rebased and upstreamed; patches are kept as reviewable series.
4. **Preserve rollback.** Releases are immutable; rollback assets are retained.