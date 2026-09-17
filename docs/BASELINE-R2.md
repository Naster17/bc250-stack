# Baseline R2 (pre-port state, 2026-09-17)

Board: r2 kernel `7.1.5-bc250rocm-r2`, params `3/0/3/0/0`, simd_count=80,
r1.2 runtime image, `HSA_ENABLE_SDMA=1`. All numbers are the reference
for the gfx1013 porting work that follows.

## Speed

rocBLAS SGEMM, native Tensile gfx1013 kernels (two reps; rep1 cold clocks,
rep2 steady state):

| N | rep1 ms / GFLOPS | rep2 ms / GFLOPS |
|---|---|---|
| 256 | 0.075 / 445 | 0.039 / 862 |
| 512 | 0.172 / 1559 | 0.089 / 3027 |
| 1024 | 0.834 / 2573 | 0.438 / 4901 |
| 2048 | 5.707 / 3010 | 2.951 / 5821 |
| 4096 | 23.370 / 5881 | 23.306 / 5897 |

Torch `a@b` fp32 (same silicon, hipBLAS path): 436 / 1543 / 2557 /
2994 / 5844 GFLOPS — within noise of native, no wrapper overhead.

Torch conv2d fp32 (MIOpen generic kernels, no perf DB):
`1x16x16x16->16`: 0.029 ms / 41 GFLOPS;
`8x32x32x32->64`: 0.068 ms / 4444 GFLOPS;
`16x64x56x56->128`: 0.812 ms / 9115 GFLOPS.

MIOpen standalone gate: Winograd algo 3, 0.024 ms, vs double CPU ref.

## Power (rocm-smi + hwmon, socket package)

- Idle: 43.1 W.
- Sustained 2048 SGEMM loop: 120–136 W.
- Post-load idle: 43.2 W (no stuck clocks).

## Quality (all PASS on r2)

- `ROCBLAS_SGEMM_VERIFY_OK` (n=64), SGEMM bench max_rel_err `0.00e+00`
  at N=256; torch 64x64 vs CPU max_abs_err `9.54e-06`.
- `ROCFFT_C2C_VERIFY_OK`, `HIPRAND_UNIF_VERIFY_OK` (mean 0.500409),
  `ROCSPARSE_CSRMV_VERIFY_OK`, `HIPSOLVER_LU_VERIFY_OK`,
  `MIOPEN_CONV_VERIFY_OK`.
- MNIST CNN (2 conv + BN + dropout, Adam, 5 epochs): loss
  `0.3883 -> 0.0722`, test acc `0.9777 -> 0.9865` (peak 0.9890).
- 50-step MLP train bit-identical GPU vs CPU
  (`1.029231 / 1.004733 / 0.985372`).

## Known gaps (port targets, in order)

1. MIOpen `gfx1013_20.HIP.fdb.txt` perf DB missing + CK grouped-conv lib
   absent → generic kernels only (conv still correct, see above).
2. hipSPARSELt: not packaged (torch tolerates absence).
3. hipTensor / RPP / rocALUTION: not built.
4. rocWMMA / CK: RDNA1 has no matrix cores — emulation-only at best,
   likely documented as not-portable after survey.
