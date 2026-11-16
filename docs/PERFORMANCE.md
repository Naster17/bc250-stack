# Measured training performance

A **bounded steady-state training-performance measurement** of fixed owned
workloads on the native gfx1013 provider. This is a bounded measurement of **tiny
owned random-init models**, not a production-speed, model-quality, or general
two-board speedup claim.

## Methodology

- **Fixed workload**: batch 2 × sequence 32 (64 tokens/step), AdamW **lr=0** holds
  weights fixed during timing (optimizer state still updates). Sequence is capped
  at the smallest admitted context.
- **AMP pattern**: FP32 master weights + autocast compute in the target dtype;
  GradScaler for FP16.
- **Timing**: cold first step recorded separately; 3 warmup steps; then 5
  synchronized intervals of 3 steps each → per-step ms summarized by **median +
  IQR**; `tokens/s = tokens × 1000 / median_ms`.
- **Numerical checks**: finite first/timed loss, no AMP overflow skip, and loss
  stability (valid because lr=0).
- **Memory**: `torch.cuda.max_memory_allocated()` after reset.
- **Utilization**: `gpu_busy_percent` returns **EOPNOTSUPP** on this board, so
  direct GPU utilization is unavailable; temperature and the DPM governor are
  recorded instead.

## Baseline results (tiny owned models)

torch 2.13.0a0+gfx1013, `gfx1013:xnack-`. Step median ms / tokens·s⁻¹:

| Model (params) | Backend | FP32 | BF16 | FP16 |
| --- | --- | --- | --- | --- |
| gpt2-tiny (21280) | eager | 7.48 / 8556 | 8.72 / 7341 | 8.98 / 7131 |
| gpt2-tiny | sdpa | 7.65 / 8363 | 9.02 / 7099 | 9.05 / 7071 |
| qwen3-tiny (312320) | sdpa | 11.23 / 5697 | 12.98 / 4931 | 13.27 / 4825 |
| qwen3-tiny | tiled | 11.04 / 5799 | 12.51 / 5117 | 12.85 / 4979 |
| qwen35-hybrid-tiny (109684) | sdpa | 35.74 / 1791 | 38.24 / 1674 | 39.04 / 1640 |

Peak allocated memory ~67–73 MB across cases. IQR 0.4–4.3% (stable).

## Bottlenecks (recorded before optimizing)

1. **Low precision is slower on these tiny models.** BF16/FP16 AMP steps are
   ~10–20% slower than FP32. The models are memory-bound and tiny, so autocast
   cast overhead dominates and there is no compute win.
2. **Native tiled ≈ math SDPA at this size** — competitive, not a large win on
   tiny shapes.
3. **Hybrid gated-delta/conv reference path is the dominant cost.** The hybrid is
   ~3–5× slower than gpt2/qwen3 because the frozen environment runs the reference
   (torch) gated-delta/conv path; the native op overlay is the acceleration route.
4. **Large first-step JIT cost for the hybrid** (Triton JIT/autotune) — an
   admission/warmup cost, not steady-state throughput.

## Native opt-in optimizations (measured gains)

All are **opt-in**; plain defaults are bitwise unchanged. Bounded owned results,
not production or general-model speed guarantees.

### Gated-delta routing

Routing the hybrid's chunk call from the reference torch fallback (a many-iteration
Python triangular-inverse loop) to the native batched-solve op:

| Measure | Frozen | Routed | Speedup |
| --- | ---: | ---: | ---: |
| Hybrid model step FP32 | 36.99 ms | 15.25 ms | **2.43×** |
| Hybrid model step BF16 | 39.53 ms | 17.89 ms | 2.21× |
| Hybrid model step FP16 | 39.73 ms | 18.00 ms | 2.21× |
| Chunk op fwd+grads FP32 (tiny) | 25.49 ms | 5.88 ms | 4.33× |

Parity within pre-declared bounds plus an independent CPU/FP64 gradient anchor;
full admission battery on the routed model passed.

### Fused native solve/scan

An opt-in fused in-register triangular inverse + column-blocked linear-recurrence
scan in the native gated-delta op (selective from two chunks up; single-chunk calls
keep the default path):

| Measure | Default | Fused | Speedup |
| --- | ---: | ---: | ---: |
| Chunk op fwd+grads FP32 (B1 H16 D128 L1024) | 26.17 ms | 14.49 ms | **1.81×** |
| Chunk op fwd+grads BF16/FP16 (same) | 26.76/26.84 ms | 15.16/15.14 ms | 1.76/1.77× |
| Hybrid model step FP32 (seq 256, routed + fused) | 19.05 ms | 16.55 ms | **1.15×** |

The fused path is closer to the FP64 reference than the frozen fallback; the
default path is bitwise identical to the installed module.

### Native causal-conv route

Routing the Transformers `causal_conv1d_fn` reference to the installed
trainable-weight native conv (with a value-identical contiguous copy to fix an
uncoalesced strided input):

| Case | Reference | Conv route | Conv + fused delta |
| --- | ---: | ---: | ---: |
| LoRA 1024 (steady s/update) | 36.29 | 35.77 (**1.0145×**) | 35.29 (**1.0283×**) |
| LoRA 4096 (cold s/update) | 326.11 | 320.11 (**1.0187×**) | — |

The step win is bounded by conv's small share and grows with sequence length; the
dominant long-sequence cost is the structural GEMM term, not conv.

## The structural GEMM term

On the real LoRA workload the step is **GEMM-compute-bound**, not launch-bound:
the GPU is ~100% busy and ~90% of device time is in Tensile FP16 GEMMs (the large
base projections, not the cheap LoRA adapters). The per-operation speed frontier
is exhausted — the GEMM term is structural and the native fused non-GEMM levers
are near-optimal. HIP graph capture is bitwise-exact but bounded by the small
non-GEMM share on the real model. Further step-time reduction needs a
model/algorithm-level approach, not per-kernel tuning.

## Scope

Single-board bounded measurement; two boards buy capacity, not a linear speedup.
Optimizing the remaining bottlenecks (low-precision cast overhead, larger-geometry
admission, a layout-aware conv kernel) is separate future work. Eager first-order
only; the opt-in routes carry no default/release change.