# Phase 4: minimal HIP-only MIOpen for gfx1013 - 2026-09-13

Source: `rocm-libraries-gfx1013` @ `b578bcd` + 6 one-line arch patches.
Container: `bc250-rocm-build-tools:7.2.4`, GPU-free, network-free.

## Research outcome (go)

MIOpen has no gfx1013 support upstream, but nothing structural blocks it:
device identity flows from `hipDeviceProp_t.gcnArchName` (reads
`gfx1013:xnack-` on this board), `GetDeviceNameFromMap` passes unknown
names through, and HIP-direct convolution solvers gate on
`StartsWith(name, "gfx103")` family checks. gfx101x-specific paths are
only two: an fp16 winograd solver (covers gfx1011/1012) and no RDNA1
gfx10 support in asm/CK kernels (correctly left out).

## Patches (6 lines, in the lab source tree)

- `miopen/src/target_properties.cpp`: `{"gfx1013", "gfx1013"}` map
  entry (keeps the native name instead of falling through).
- 5 solver gates extended with `|| (name == "gfx1013")`:
  `conv_hip_dir2Dfwd`, `conv_hip_dir2D11x11`, `conv_hip_dir2D_bwdWrW_53`
  (HIP direct), `conv_winoRxS`, `conv_winoRxS_fused` (Winograd).
- Deliberately NOT patched: asm implicit-GEMM and CK solvers (they
  need XDL/WMMA hardware gfx1013 lacks). They stay inapplicable.

Script: `/tmp/patch-miopen-gates.py` (kept on the board; the source
tree itself is a git checkout so the diff is auditable).

## Build configuration

- `MIOPEN_BACKEND=HIP`, `MIOPEN_STANDALONE_BUILD=ON` with all
  FetchContent sources pre-staged via `FETCHCONTENT_SOURCE_DIR_*`
  (offline-clean): bzip2 1.0.8, sqlite 3.51.3, eigen 3.4.0,
  nlohmann_json 3.11.3, FunctionalPlus 0.2.25, frugally-deep 0.15.31,
  googletest 1.17.0 (all sha256-verified, kept in
  `~/bc250work/third-party/`).
- `MIOPEN_USE_COMPOSABLEKERNEL=OFF`, `MIOPEN_USE_HIPBLASLT=OFF`,
  `MIOPEN_USE_MLIR=OFF`, `MIOPEN_USE_ROCTRACER=OFF`,
  `MIOPEN_ENABLE_AI_KERNEL_TUNING=OFF`,
  `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF`, `MIOPEN_BUILD_DRIVER=OFF`,
  `BUILD_TESTING=OFF`, `MIOPEN_INSTALL_GPU_DATABASES=gfx1030`.
- `COMGR=ON` + `HIPRTC=ON` (must be equal by MIOpen's own check);
  rocBLAS/rocPRIM/rocRAND from our native prefixes; half headers
  staged at `third-party/half/include/half/half.hpp`.
- `CMAKE_CXX_FLAGS=-Wno-error=missing-noreturn`: newer amdclang flags
  pre-existing missing-noreturn in pooling solver headers; upstream
  code issue, demoted not patched.
- Installed: `~/bc250work/miopen-build-gfx1013/miopen-prefix/`,
  `libMIOpen.so.1.0` (452 MiB). Needs at runtime: hiprtc, amd_comgr,
  rocblas, sqlite3, amdhip64 (all in the base image or our prefixes).

Build script: `build-miopen-gfx1013.sh`. Dep builders:
`build-miopen-deps.sh` (json+eigen), `build-miopen-sysdeps.sh`
(sqlite+bz2 static, superseded by standalone wrappers but kept).

## Gate (r1 kernel, HSA_ENABLE_SDMA=0, CPU-verified)

`MIOPEN_CONV_VERIFY_OK`: 1x16x16x16 fp32 input, 16x3x3 conv stride 1
pad 1, full double-precision CPU reference, tolerance 2e-3.
Find selected Winograd (fwd algo id 3, 0.024 ms, ws 0) - the patched
solver dispatch works end to end: device name -> target properties ->
solver applicable -> kernel compiled via comgr/hiprtc -> correct result.
Two benign warnings: missing `gfx1013_20.HIP.fdb.txt` (no shipped
tuning DB for this arch, expected) and missing CK grouped-conv lib
(CK disabled, expected).

Log: `results/miopen-conv-gfx1013.log`. Program: `miopen-conv-verify.cpp`.
# Phase 2 libraries (level 1): headers, hipFFT, rocRAND - 2026-09-13

Source: `rocm-libraries-gfx1013` @ `b578bcd` (same tree as rocBLAS).
Container: `bc250-rocm-build-tools:7.2.4`, GPU-free, network-free.
Compilers: `/opt/rocm/bin/amdclang{,++}` (hipCUB refuses plain g++).
Extra CMake hints: `ROCmCMakeBuildTools_DIR`, `ROCM_DIR=/opt/rocm`,
`HIP_DIR=/opt/rocm/lib/cmake/hip`, `CMAKE_PREFIX_PATH=/opt/rocm`.

## Built

- `~/bc250work/rocm-headers-gfx1013/`: `rocprim-prefix`,
  `hipcub-prefix`, `rocthrust-prefix` (header-only + cmake packages).
- `~/bc250work/hipfft-build-gfx1013/hipfft-prefix`: hipFFT wrapper
  against the native rocFFT prefix (`rocfft_DIR`).
- `~/bc250work/rocrand-build-gfx1013/`: `rocrand-prefix`
  (`AMDGPU_TARGETS=gfx1013`), `hiprand-prefix` (against it).

Build scripts: `build-headers-gfx1013.sh`, `build-hipfft-gfx1013.sh`,
`build-rocrand-gfx1013.sh`.

## Gates (r1 kernel, HSA_ENABLE_SDMA=0, all CPU-verified)

- `HIPFFT_C2C_VERIFY_OK length=16` - hipFFT forward c2c vs CPU DFT.
- `HIPRAND_UNIF_VERIFY_OK n=1048576 mean=0.500409` - same-seed
  determinism (exact), range [0,1), mean within 5e-3 of 0.5.
- `ROCFFT_C2C_VERIFY_OK length=16 iterations=1` and
  `length=64 iterations=8` - re-run after the fix below.

Runner: `run-phase2-gate-guarded.sh`.

## Test-harness bug found and fixed (same session)

`rocfft-c2c-verify.cpp` and the new `hipfft-c2c-verify.cpp` shared an
input expression with unsigned wraparound:
`(i * 3) % 7 - 3` on `size_t i` wraps to ~2^64 whenever the remainder
is below 3, so half the "inputs" were ~1.8e19 and the CPU reference
itself was garbage. The GPU transforms were correct throughout (device
output matched the garbage reference to ~1e-7 relative). Fixed with an
explicit `static_cast<int>` before subtracting; both gates pass after
the fix. The prior session's rocFFT gate record is superseded by the
re-runs above. Lesson: print the CPU reference before trusting a
mismatch - done via `diag-hipfft.cpp` (kept in the lab).
# Phase 3 libraries: rocSPARSE/hipSPARSE, rocSOLVER/hipSOLVER - 2026-09-13

Source: `rocm-libraries-gfx1013` @ `b578bcd`.
Container: `bc250-rocm-build-tools:7.2.4`, GPU-free, network-free.
Compilers: `/opt/rocm/bin/amdclang{,++}`.

## Built

- `~/bc250work/rocsparse-build-gfx1013/`: `rocsparse-prefix`
  (`GPU_TARGETS=gfx1013:xnack-`, `BUILD_WITH_ROCBLAS=ON` against the
  native rocBLAS), `hipsparse-prefix` (clients off).
- `~/bc250work/rocsolver-build-gfx1013/`: `rocsolver-prefix`
  (`AMDGPU_TARGETS=gfx1013:xnack-`, `BUILD_WITH_SPARSE=ON`, system
  LAPACK in module mode, fmt 11.1.4 from `~/bc250work/fmt-build/`),
  `hipsolver-prefix` (sparse off, fortran off, internal OpenBLAS
  build off, system OpenBLAS in module mode).
- `~/bc250work/fmt-build/fmt-prefix`: fmt 11.1.4, source fetched on
  the host (zip sha256
  `49b03960...e213d3e6`), built offline. Kept: rocSOLVER links it.
- `~/bc250work/syslibs-fallback/`: `libopenblas.so.0` +
  `libgfortran.so.5` extracted from the build-tools image for gates.
  The r1.2 runtime image must ship these (libhipsolver DT_NEEDED).

Build scripts: `build-fmt.sh`, `build-rocsparse-gfx1013.sh`,
`build-rocsolver-gfx1013.sh` (rocsolver step skipped if installed).

## Gates (r1 kernel, HSA_ENABLE_SDMA=0, all CPU-verified)

- `ROCSPARSE_CSRMV_VERIFY_OK` - scsrmv on a 4x4/7-nnz matrix, exact
  element match.
- `HIPSOLVER_LU_VERIFY_OK` - Spotrf Cholesky with L*L^T
  reconstruction (err < 1e-4) plus Sgetrf LU with pivot-aware
  P*A == L*U reconstruction, through hipSOLVER -> rocSOLVER ->
  native rocBLAS/rocSPARSE.

## Findings

- `strings`-based arch detection is unreliable for these libs: the
  `gfx1010/1011/1012` strings in librocsparse come from compiler
  device tables, and `1013` hits in mangled kernel names are template
  parameters, not arch tags. The runtime gate is the proof, and it
  passes. `build.ninja` confirms `--offload-arch=gfx1013:xnack-`
  on every compile rule.
- hipSOLVER's `HIPSOLVER_INTERNAL_LAPACK_BUILD=ON` clones OpenBLAS
  from GitHub: unusable offline, set OFF. System OpenBLAS covers it.
- hipSOLVER (like cuSOLVER) needs explicit workspace buffers:
  `*_bufferSize` query + device work buffer. The verify program shows
  the pattern.
- rocSOLVER library does NOT link LAPACK (only its clients do);
  `librocsolver.so` needs just rocsparse/rocblas/amdhip64.
