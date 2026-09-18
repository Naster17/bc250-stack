# Torch capability matrix (2026-09-17/18)

Script: `bc250-stack/pytorch/torch-matrix.py` (run unbuffered: `python -u`).
Covers every supported path: cuda/arch, fp32+fp16+bf16 GEMM vs fp64 CPU
ref, conv2d fwd+bwd (tuned DB), FFT roundtrip vs CPU, RNG stats, sparse
CSRmv vs dense, `linalg.solve` + `lu_factor`, Adam train with clip,
fp16 autocast. Exit nonzero on any FAIL.

## .105 (r2 + r1.4, full timings) — MATRIX_PASS

```text
torch 2.9.1a0 cuda=True ndev=1 arch=['gfx1013']
PASS cuda-available
PASS arch-gfx1013
PASS gemm-float32 rel_err=6.69e-06 ms=0.050
PASS gemm-float16 rel_err=4.82e-04 ms=0.032
PASS gemm-bfloat16 rel_err=3.88e-03 ms=0.024
PASS conv-bwd-1x16x16-16 ms=0.033
PASS conv-bwd-8x32x32-64 ms=0.069
PASS fft-roundtrip abs_err=1.19e-06 ms=0.120
PASS rand-uniform mean=0.5000 var=0.0833
PASS sparse-csrmv abs_err=0.00e+00
PASS linalg-solve rel_err=8.94e-07
PASS lu-factor
PASS adam-train 0.9811->0.9353
PASS autocast-f16
MATRIX_PASS
```

(The two MIOpen warnings — missing shipped perf DB, missing CK grouped
lib — are benign: Find falls back to generic kernels, results correct.)

## .104 (peer, system 7.0.10-n17 image, pass/fail only)

```text
PASS cuda-available
PASS arch-gfx1013
PASS gemm-float32 rel_err=6.69e-06
PASS gemm-float16 rel_err=4.82e-04
PASS gemm-bfloat16 rel_err=3.88e-03
PASS conv-bwd-1x16x16-16
PASS conv-bwd-8x32x32-64
FAIL fft-roundtrip (isolated, see below)
```

The FFT section crashes the process
(`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`, `hipErrorLaunchFailure`),
so rand/sparse/solver/train/autocast after it never ran on .104.

## .104 FFT finding (peer-only, .105 unaffected)

- Isolated `torch.fft.fft2` on 4x256x256 faults;
  size sweep: n=8 OK, larger (32x32 factors 8_4x8_4) illegal instruction.
- `.105` runs the identical 256x256 FFT clean (`abs_err=1.19e-06`).
- `.104` differences: stock system image `7.0.10-0-n17` (no PASID/runlist
  TLB params), but `simd_count=80` and same r1.4 userspace.
- Console also shows `rocfft_rtc_helper: libamdhip64.so.7 not found`,
  i.e. RTC fallback kernels are broken in this mount set too.
- Verdict: environment/firmware difference on the peer, not a torch or
  rocFFT regression — precompiled n=8 path works, RTC path faults.
  Action: re-run the peer matrix after booting the peer into r2;
  until then peer FFT is marked KNOWN-FAIL.
