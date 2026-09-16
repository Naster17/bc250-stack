# Native gfx1013 PyTorch 2.9.1 - 2026-09-16

## Build isolation

- Source: `~/bc250work/pytorch-2.9.1` (v2.9.1 tree, `.git` stripped at stage).
- Build container image: `bc250-rocm-build-tools:7.2.4`, GPU-free, network-free.
- Script: `bc250-rocm-lab/build-torch-gfx1013.sh`.
- Eigen 3.4.0 copied into `third_party/eigen`; all other submodules pre-staged.
- Patches `0004`-`0011` applied inside the container after stage copy:
  - `0004`: forward `Python_*` hints from setup.py env into CMake configure.
  - `0005`: roctx/roctracer prefix fallback in `LoadHIP.cmake`.
  - `0006`: link `roc::hipblaslt` when found (`Dependencies.cmake`).
  - `0007`: add roctracer prefix include to `torch_hip`.
  - `0009`: torch_python roctracer include + dynamo C sources as LANGUAGE C
    with `-std=gnu11`, plus unconditional `pycore_pystate.h` include.
  - `0010`: functorch `.c` sources as LANGUAGE C with `-std=gnu11`.
  - `0011`: host `CMAKE_CXX_FLAGS += -fclang-abi-compat=17` to match HIP host
    mangling (both compilers are amdclang++ here).
- Python dev overlay: `/opt/pydev` (headers + `libpython3.12.a`), plus
  multiarch `pyconfig.h` at both `include/x86_64-linux-gnu/...` and
  `include/python3.12/x86_64-linux-gnu/...` paths.
- `CFLAGS/CPPFLAGS` carry the pydev + numpy include dirs for the stub compile.

## Native prefixes consumed

- rocBLAS/hipBLAS/hipblas-common/hipblaslt/roctracer/rocFFT/hipFFT/rocRAND/
  hiprand/rocSPARSE/hipSPARSE/rocSOLVER/hipSOLVER/rocPRIM/hipCUB/rocThrust/
  MIOpen, all native gfx1013 builds, plus LLVM `libomp.so` for the runtime.
- hipBLAS was rebuilt `BUILD_WITH_SOLVER=ON` against the native rocsolver
  prefix (`rebuild-hipblas-solver-gfx1013.sh`); the wrapper now exports
  `hipblasSgetrsBatched` and family (24 getrs symbols).
- hipblaslt 7.2.4 headers + `libhipblaslt.so.1.2.70204` from the upstream deb;
  roctracer headers + `libroctx64.so.4.1.70204` likewise.
- Upstream hipblaslt cmake configs reused; local `hipblaslt-config-version.cmake`
  shim only.

## Configure gates passed

- `find_package(Python COMPONENTS Development.Module NumPy)` OK (3.12.3).
- `hipblaslt VERSION: 1.2.2.70204` found via prefix; outer-vec/vec-ext
  try-compiles fail only on missing include in the probe (headers resolve in
  the real target via `roc::hipblaslt`, benign).
- `ROCM_ROCTX_LIB=/roctracer/lib/libroctx64.so`.
- `Configuring done`, `Generating done`, `build.ninja` written.

## Wheel

- `torch-2.9.1a0+gitunknown-cp312-cp312-linux_x86_64.whl`, 133,327,089 bytes,
  sha256 `abe93cd86b33a595b52f290e050a925fc4ac04e4746fb8f7f9bcefdfb8aa08bc`.
- Built via `build-wheel-gfx1013.sh` (same mounts + `USE_SYSTEM_LIBS=1`,
  nccl/distributed off, CFLAGS for `stub.c`).

## GPU validation (r1 kernel, r1.1 runtime image + native mounts)

- `torch.__version__` 2.9.1a0; `cuda.is_available()` True; device_count 1;
  arch list `['gfx1013']`.
- 64x64 fp32 GEMM on cuda OK.
- 50-step Linear MLP train loop, SGD lr=0.01, batch 128: GPU and CPU losses
  bit-identical at steps 0/24/49 (`1.029231 / 1.004733 / 0.985372`), TRAIN_OK.
- Runtime needs: `LD_PRELOAD=<omp>/libomp.so` (LLVM libomp, sha256
  `874e24cdfe2c07b8d9bde7f2723983b7d0123917f10697204d6f77d4ba9e1055`),
  `LD_LIBRARY_PATH` with syslibs (openblas/gfortran) + all native prefixes,
  `HSA_ENABLE_SDMA=1`, `OMP_NUM_THREADS=4`.
