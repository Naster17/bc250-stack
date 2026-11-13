# PLAN-10 — measured training performance (D64 + D65)

Completed 2026-11-12. PLAN-10 replaces correctness-only evidence with a **bounded
steady-state training-performance measurement** of a fixed owned workload on the
native gfx1013 provider. Deliverables: the owned benchmark fixture
`build_tools/bc250/benchmark_training_d64.py` (D64) and a qualified GPU run on the
.104 release (D65). Evidence: `kernel/artifacts/plan10-d65/`.

This is a bounded measurement of **tiny owned random-init models**, not a
production-speed, model-quality, or general two-board speedup claim.

## Methodology

Reuses the D62 owned model registry/builders and the qualified timing protocol:
- **Fixed workload**: batch 2 × sequence 32 (64 tokens/step), AdamW **lr=0** holds
  weights fixed during timing (optimizer state still updates). Sequence is capped
  at the smallest admitted context (gpt2-tiny `n_positions=32`).
- **AMP pattern**: FP32 master weights + autocast compute in the target dtype;
  GradScaler for FP16. (Casting params to FP16 produces FP16 grads GradScaler
  refuses — see defects.)
- **Timing**: cold first step recorded separately; 3 warmup steps; then 5
  synchronized intervals of 3 steps each → per-step ms summarized by **median +
  IQR**; `tokens/s = tokens × 1000 / median_ms`.
- **Numerical checks**: finite first/timed loss, no AMP overflow skip
  (`scaler.get_scale()` not decreased), and loss stability (`assert_close` to the
  first loss, rtol/atol 1e-4, valid because lr=0).
- **Memory**: `torch.cuda.max_memory_allocated()` after reset.
- **Utilization**: `gpu_busy_percent` returns **EOPNOTSUPP** on this board, so
  direct GPU utilization is unavailable; the report records that boundary plus
  hwmon temperature (53→54 °C across the run) and the DPM governor (`auto`).

## Results — .104 release, 15/15 cases passed

torch 2.13.0a0+gfx1013, hip 7.17.0, transformers 5.2.0 / accelerate 1.12.0 /
peft 0.18.1, `gfx1013:xnack-`. Fixture sha256 `b4adb7f9…` bound into the report.
Step median ms / tokens·s⁻¹ (IQR% in the JSON):

| Model (params) | Backend | FP32 | BF16 | FP16 |
| --- | --- | --- | --- | --- |
| gpt2-tiny (21280) | eager | 7.48 / 8556 | 8.72 / 7341 | 8.98 / 7131 |
| gpt2-tiny | sdpa | 7.65 / 8363 | 9.02 / 7099 | 9.05 / 7071 |
| qwen3-tiny (312320) | sdpa | 11.23 / 5697 | 12.98 / 4931 | 13.27 / 4825 |
| qwen3-tiny | tiled | 11.04 / 5799 | 12.51 / 5117 | 12.85 / 4979 |
| qwen35-hybrid-tiny (109684) | sdpa | 35.74 / 1791 | 38.24 / 1674 | 39.04 / 1640 |

Peak allocated memory ~67–73 MB across cases. IQR 0.4–4.3% (stable). Board clean:
0 containers before/after; no source/provider/prefix/default change, no reboot.

## Bottlenecks and regressions (recorded before optimizing)

1. **Low precision is slower on these tiny models.** BF16/FP16 AMP steps are
   ~10–20% slower than FP32 (e.g. gpt2 eager 7.48 → 8.72/8.98 ms). The models are
   memory-bound and tiny, so autocast cast overhead dominates and there is no
   compute win. Low precision is not a speedup here.
2. **Native tiled ≈ math SDPA at this size.** qwen3 tiled is marginally faster
   than sdpa (FP32 11.04 vs 11.23 ms; BF16 12.51 vs 12.98 ms) — competitive, not a
   large win on tiny shapes.
3. **Hybrid gated-delta/conv reference path is the dominant cost.** qwen35-hybrid
   is ~3–5× slower than gpt2/qwen3 (35.7 ms vs 7.5/11.2 ms FP32) because the
   frozen 5.2.0 environment runs the reference (torch) gated-delta/conv path; the
   native op overlay (5.19.0-gated) is the acceleration route, not present here.
4. **Large first-step JIT cost for the hybrid.** qwen35-hybrid BF16/FP16 cold
   first step is ~16.7 s (Triton JIT/autotune) vs ~38 ms steady-state; FP32 first
   step ~0.43 s. This is an admission/warmup cost, not steady-state throughput.

## Defects found and fixed while qualifying the fixture

1. **Sequence overflow segfault**: the initial SEQ=64 exceeded gpt2-tiny
   `n_positions=32`, indexing the position embedding out of bounds and faulting
   the GPU (exit 139). Fixed to SEQ=32; added incremental per-case report writes
   so a fault preserves completed measurements.
2. **FP16 GradScaler refusal**: casting model params to FP16 produced FP16 grads
   that `scaler.unscale_` refuses. Fixed to the qualified AMP pattern (FP32 master
   weights + autocast compute).

## Verification

- 8 new local tests (`bc250_benchmark_d64_test.py`); full bc250 suite **198 passed
  / 84 subtests**; Black clean (ruff unavailable on the workstation).
- D65: 15/15 cases passed; fixture hash bound into the report; pre/post container
  counts 0; temperature stable; run via the ordinary `bc250-train` caller under
  the immutable release (no mutable runtime lock).

## Scope and remaining

PLAN-10 is complete for its bounded scope: a fixed owned workload measured with
warmup/repeated intervals, tokens/s, step time, memory, numerical checks, and an
honest utilization boundary; FP32/BF16/FP16 and eager/sdpa/tiled compared;
bottlenecks recorded before any optimization. **Single-board** scope; a distributed
(two-board) performance benchmark is a follow-up (L01 was correctness-first, not a
speedup claim). Optimizing the recorded bottlenecks (hybrid native-op overlay,
low-precision cast overhead) is separate future work, not part of this measurement.

With PLAN-10 done, **all ten original work packages are complete**. Publication of
the release still requires explicit user instruction.

---

# D66 follow-up — native gated-delta routing (2026-11-13)

The first bounded optimization scope for PLAN-10 bottleneck #3 (the hybrid
gated-delta reference path). Root cause: the frozen Transformers 5.2.0 torch
fallback `torch_chunk_gated_delta_rule` builds its unit-lower-triangular inverse
with a **63-iteration Python loop** — hundreds of sequential small kernel
launches per call, which dominates the launch-bound tiny benchmark geometry.
The qualified native `torch.nn.bc250_gated_delta.chunk_gated_delta_rule`
(A06 preparation + A07 low-precision parity) uses batched triangular solves.

An explicit **opt-in** helper (`route_qwen35_gated_delta_d66.py`,
`bc250-therock` fork) routes the model's chunk call to the native op under
strict gates (Transformers 5.2.0, `qwen3_5_text`, actual gfx1013), with a kill
switch (`BC250_NATIVE_GATED_DELTA=0`), counted fallbacks and exact uninstall.
The trainable-weight causal conv and the recurrent/decode path are not routed;
no default, package, source or release identity changes.

## Results (.104 release, owned qwen35-hybrid-tiny, D64 protocol)

| Measure | Frozen | Routed | Speedup |
| --- | ---: | ---: | ---: |
| Model step FP32 | 36.994 ms (1730 tok/s) | 15.254 ms (4196 tok/s) | **2.425×** |
| Model step BF16 | 39.529 ms | 17.894 ms | 2.209× |
| Model step FP16 | 39.728 ms | 18.002 ms | 2.207× |
| Chunk op fwd+all-grads FP32, B2 L32 H2 D16 | 25.49 ms | 5.88 ms | 4.33× |
| Chunk op fwd+all-grads FP32, B2 L256 H2 D16 | 31.20 ms | 8.70 ms | 3.59× |

IQR ≤ 1.5%. Steady-state breakdown: chunk-op forward 8.02 ms of a 34.93 ms step
(23%); forward+backward ≈ 25.5 ms ≈ 73% of the step; conv forward 0.19 ms. The
op-level win shrinks with sequence length because the remaining ATen solve/scan
compute grows (A06 measured ~7% at B1 H16 D128 L1024).

## Qualification

- Parity within pre-declared bounds: FP32 loss Δ0, logits rel-L2 1.5e-7, worst
  gradient rel-L2 8.4e-4 (gate parameters `A_log`/`dt_bias`); BF16/FP16 within
  the A06/A07-style 0.04 gradient / 0.015 loss bounds. Op-level FP32 ≤ 3.1e-6.
- **Independent CPU/FP64 gradient anchor:** both GPU paths within 1.2e-3 rel-L2
  of an exact-weight CPU float64 reference (bound 5e-3) — rounding divergence,
  not a structural defect.
- The original FP32 model-gradient bound (1e-4) **failed** in an intermediate
  run at 8.43e-4 and is retained on record; the revised 2e-3 bound was
  pre-declared with justification before the confirmation runs.
- Full D62 admission battery on the routed model passed (forward vs CPU/FP64,
  backward, optimizer step, bitwise checkpoint save/resume, exact continuation);
  five refusal checks passed (kill switch, wrong model, CPU, uninstall identity,
  initial-state fallback).
- Board clean: 0 containers/GPU clients pre/post, temperature stable, no reboot;
  ordinary `bc250-train` caller under the immutable release. 21 new local tests
  (suite 225), Black clean.

## Boundaries and remaining headroom

Bounded owned tiny-model result — not a production, model-quality, convergence
or general-model speed guarantee. Eager first-order only; no compile/graph or
higher-order gradients. PLAN-10 bottlenecks #1 (low-precision cast overhead) and
#4 (cold JIT first step) are unchanged. Remaining optional scopes: fused native
solve/scan kernels (done in D68, below), trainable-weight native conv,
low-precision cast overhead, larger-geometry admission. Full report and evidence
hashes: `bc250-therock` fork, `docs/development/bc250_bottleneck_d66_results/`.

---

# D68 follow-up — fused native solve/scan (2026-11-13)

The first bounded scope of the deeper bottleneck work after D66. Section
attribution of the native chunk op (owned profile fixture, exact-replica
CUDA-event breakdown + isolated microbenchmarks): the sequential Python chunk
scan is **~70% of forward+backward** at B1/H16/D128/L1024 (launch-bound
autograd through the loop), and the two batched ATen triangular solves ~48% at
the tiny single-chunk geometry.

The patched Torch source (fork commit `262accc`, patch `0023` of the complete
series) adds an opt-in `use_fused_solve_scan=True` to the native op:

- a fused in-register Triton unit-triangular inverse with analytic backward
  (`grad_A = -tril(M^T dM M^T, -1)`) replaces both solves;
- a fused column-blocked linear-recurrence scan replaces the loop:
  `S_{i+1} = (d_i I - k_i^T w_i) S_i + k_i^T u_i`,
  `out_i = (q_i - attn_i w_i) S_i + attn_i u_i`, with an adjoint backward and
  saved entering states; the M/N/P/R precompute stays autograd-visible ATen;
- elementwise Triton convention (no `tl.dot`, wave32, `num_warps=4`,
  `enable_fp_fusion=False`), matching A05/A06;
- **selective from two chunks up** (`FUSED_SCAN_MIN_CHUNKS = 2`): the measured
  decomposition showed every fusion variant regresses at one chunk
  (0.79–0.94×) and wins from two chunks up, so single-chunk calls keep the
  default path; explicit refusal above a 256 MiB saved-state cap.

## Results (.104 release, final qualification run)

| Measure | Default | Fused | Speedup |
| --- | ---: | ---: | ---: |
| Chunk op fwd+all-grads FP32, B1 H16 D128 L1024 | 26.17 ms | 14.49 ms | **1.81×** |
| Chunk op fwd+all-grads BF16/FP16, same geometry | 26.76/26.84 ms | 15.16/15.14 ms | 1.76/1.77× |
| Chunk op fwd+all-grads FP32, B2 H2 D16 L256 | 8.09 ms | 5.12 ms | 1.58× |
| Chunk op fwd+all-grads FP32, B2 H2 D16 L32 (C=1) | 4.57 ms | 4.30 ms | 1.0× (default) |
| Hybrid-tiny model step FP32, seq 256 (D66 route + flag) | 19.05 ms | 16.55 ms | **1.15×** |
| Hybrid-tiny model step BF16/FP16, seq 256 | 22.55/22.82 ms | 19.33/19.68 ms | 1.17/1.16× |

Model step vs frozen at seq 256: 2.60/2.38/2.35× (D66 route alone: 2.26/2.04/
2.03×). At seq 32 the flag keeps the default path — no regression.

## Qualification

- Extended A06 battery on the candidate: **72/72** (fused CPU-FP64 recurrence
  with initial state + state gradients at the unchanged A06 tolerances;
  fused-vs-default equivalence; low-precision reference preparation; refusals).
- Op parity fused-vs-default: FP32 ≤ 1.5e-6, BF16/FP16 ≤ 9.3e-5 rel-L2
  (pre-declared bounds 1e-5 / 1e-2). Independent CPU/FP64 recurrence anchor at
  0.3–5.6% of the A06 tolerances.
- Model parity at seq 256 within the D66 bounds (FP32 worst gradient 1.52e-3 ≤
  2e-3); the fused path is **7× closer to the FP64 reference than the frozen
  fallback** (2.1e-4 vs 1.4e-3 rel-L2).
- Default path bitwise identical to the installed module; non-bool flag and
  state-cap refusals pass; per-case fused-kernel counters prove which path ran.
- Board clean pre/post (0 containers/GPU clients, no reboot); ordinary caller
  under the immutable release; the installed module is unchanged. 20 new local
  tests (suite 269), Black clean.
- Retained defect records: the unconditional first design regressed at one
  chunk (fixed by the measured selectivity); a fixture-side routing-injection
  leak contaminated an intermediate model-timing control (root-caused by
  inject/restore tracing, fixed, re-qualified).

## Boundaries and remaining headroom

Bounded owned result — not a production or general-model speed guarantee.
Eager first-order only; opt-in, no default/release change. Remaining optional
scopes: finer per-geometry selectivity (~5% at four chunks where scan-only
beats the combined path), trainable-weight native conv, low-precision cast
overhead, larger-geometry admission. Full report and evidence hashes:
`bc250-therock` fork, `docs/development/bc250_gated_delta_d68_results/`.