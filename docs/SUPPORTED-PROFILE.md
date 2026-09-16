# BC250 gfx1013 ROCm Supported Profile

This is an experimentally validated local profile, not AMD official ROCm support.

## Required kernel profile

- Kernel: `6.18.50-bc250rocm-test`.
- 40 CU: `amdgpu.bc250_cc_write_mode=3`.
- PASID flush: `amdgpu.bc250_flush_pasid_kiq=0`.
- Runlist map and unmap flush: `amdgpu.bc250_flush_by_runlist=3`.
- Scheduler: `amdgpu.sched_policy=0`.
- Fault policy: `amdgpu.gpu_recovery=0`.
- Normal Alpine `6.18.50-0-lts` remains the default GRUB entry. The test kernel is selected only through a one-shot entry.

## Required userspace profile

- Ubuntu 24.04 rootful Podman container with ROCm 7.2.4 userspace.
- `HSA_ENABLE_SDMA=0` remains the default for validated work; `=1` is
  validated too on the fixed firmware (see SDMA line below).
- Native gfx1013 rocBLAS prefix.
- Native hipBLAS wrapper prefix built against that rocBLAS prefix.
- Native gfx1013 rocFFT prefix with its AOT cache.
- Native llama.cpp HIP build from `~/llama-n17-gfx1013-fa-build` with the RDNA1 macro patch.
- `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` for the validated llama.cpp path.

## Validated capabilities

- HIP allocation, memcpy, dispatch, synchronization, and CPU result verification.
- Native rocBLAS FP32, FP64, and FP16 GEMM correctness gates.
- Standard hipBLAS wrapper API through native rocBLAS.
- Native rocFFT complex-forward transforms against CPU DFT references.
- Full GGML HIP backend CPU-comparison suite: 15,988 supported comparisons passed.
- Gemma 4 Q8 all-layer HIP generation with Flash Attention forced on.
- Gemma 4 Q8 perplexity gate (wikitext-2 test, ctx 2048, 4 chunks):
  PPL = 63.9747, identical at SDMA=0 and SDMA=1.
- PyTorch 2.9.1a0 gfx1013 wheel: `cuda.is_available()` True, arch
  `['gfx1013']`, 64x64 GEMM OK, 50-step MLP train bit-identical to CPU
  (`1.029231 / 1.004733 / 0.985372`), `HSA_ENABLE_SDMA=1`.
- Hardware SDMA engine: works with the navi12 microcode substitution
  carried only inside the r1 initramfs (stock `/lib/firmware`
  untouched). Full 4 KiB-16 MiB copy sweep passes CPU-verified with
  `HSA_ENABLE_SDMA=1`. Stock cyan_skillfish2 ucode hangs above 16 KiB.

## Not claimed

- AMD official support.
- General stock ROCm package compatibility; stock rocBLAS lacks native gfx1013 code objects.
- Stock hipBLAS wrapper without solver API (rebuilt locally with
  `BUILD_WITH_SOLVER=ON`; see `PYTORCH-BUILD-2026-09-16.md`).
- Upstream hipblaslt/roctracer debs (headers + libs only, no gfx1013 kernels
  claimed); every other ROCm library.
- Persistent test-kernel boot or unattended workloads.

## Primary records

- `RESULTS-2026-09-12.md`
- `ROCBLAS-BUILD-2026-09-12.md`
- `ROCFFT-BUILD-2026-09-13.md`
- `HIPBLAS-BUILD-2026-09-13.md`
- `PYTORCH-BUILD-2026-09-16.md`
- `LLAMA-HIP-BUILD-2026-09-13.md`
- `FLASH-ATTN-FIX-2026-09-13.md`
- `SDMA-FIX-2026-09-13.md`
