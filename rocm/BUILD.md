# Native gfx1013 userspace builds

Source: `boondocklabs/rocm-libraries` branch `gfx1013-cyanskillfish`
(`b578bcd`), rocm-cmake `b395920`. GPU-free, network-free containers.

- rocBLAS/hipBLAS(Tensile gfx1013 objects) -> `rocblas-prefix`,
  `hipblas-prefix`, `hipblas-common-prefix`.
- hipBLAS solver rebuild (`rebuild-hipblas-solver-gfx1013.sh`,
  `BUILD_WITH_SOLVER=ON` vs native rocsolver prefix) adds
  `getrsBatched` family; torch links it.
- rocFFT/hipFFT, rocRAND/hiprand, rocSPARSE/hipSPARSE,
  rocSOLVER/hipSOLVER (`gfx1013:xnack-`), rocPRIM/hipCUB/rocThrust headers.
- MIOpen: `miopen-gfx1013.diff` (this dir, = working-tree diff at
  `b578bcd`), minimal HIP-only standalone build; gate
  `MIOPEN_CONV_VERIFY_OK`.
- hipblaslt 7.2.4 + roctracer: upstream debs, headers+libs only.
- LLVM `libomp.so` staged from the build-tools image for the torch runtime.
