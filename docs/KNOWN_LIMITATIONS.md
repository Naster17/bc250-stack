# Known limitations

Explicit boundaries of the BC-250 gfx1013 port and the
`10.2-gfx1013-20261111-138a66f52` release. These are not bugs to hide — they are
the honest scope. The machine-readable form is
[`../sbom/known-limitations.json`](../sbom/known-limitations.json).

## Hardware

- **No MFMA/WMMA matrix instructions** — cannot be added in software; low-precision
  GEMM/dot paths use scalar/DPP lowering.
- **SDMA disabled**; experimental VM heap disabled (safe defaults).
- **12 CPUs / ~15 GiB RAM per board**, shared CPU/GPU memory. A single all-resident
  1B FSDP state floor exceeds physical RAM.
- Advanced hardware profiling (PC sampling/RDC) unqualified; attained-clock /
  temperature / power telemetry semantics unvalidated. `gpu_busy_percent` returns
  EOPNOTSUPP, so direct GPU utilization is unavailable.

## Software subsets

- Scalar backends have documented API/type/operator subsets.
- **Model admission is per-architecture, bounded evidence** — not a universal
  support claim. Admitted: causal LMs (gpt2/qwen3/qwen3.5-hybrid, D62) plus owned
  pure-torch UNet2D (diffusion-style noise-prediction) and RNN/LSTM/GRU
  (PLAN2-06), each through load → forward vs CPU/FP64 → backward → optimizer →
  bitwise resume. diffusers is absent (the UNet is owned pure-torch, not a full
  diffusion pipeline); ViT/ConvNeXt, real diffusion pipelines and larger
  geometries remain separate future admissions.
- Production nonsymmetric eig uses host LAPACK (retained; the native small-Schur
  prototype is slower for small orders, N2.6). **Native ILU(0)/IC(0) sparse
  factorization + preconditioned solve now exists** (PLAN2-03, opt-in, machine
  precision); rocSPARSE has no general direct sparse LU/Cholesky, and extreme
  conditioning/larger orders remain separate from the bounded dense-LU fallback.
- **Math SDPA qualified**; the opt-in tiled/hybrid attention is now
  performance-qualified (PLAN2-04): causal forward 1.5–2.9× faster, full
  attention 12–86× less peak memory but slower compute; dropout/compile
  boundaries remain documented refusals for the tiled path.
- **torch.compile coverage expanded** (PLAN2-05) to attention/LoRA/embed+CE/
  dynamic shapes (fullgraph, real Inductor, parity vs CPU + eager) — correctness
  coverage, not a throughput claim (tiny fixtures do not speed up). **HIP graph
  capture of a training step is qualified bitwise-exact** (PLAN2-05, opt-in):
  fwd+bwd+eager AdamW, whole-step capturable AdamW, checkpointing,
  make_graphed_callables, fresh-process recapture/resume; ~1.9× on a launch-bound
  owned stack. Boundaries: static shapes, dropout=0, graph pool not checkpointable
  (resume needs recapture); real-model wiring and compiled-optimizer/large-model
  throughput are separate. Recorded precision properties (not compile defects):
  FP16 attention backward and dynamic FP16 fused-attention lowering differ from
  eager/CPU at the ~% level.
- **W8 and native NF4/QLoRA quantization exist** (NF4 is explicit opt-in,
  PLAN2-02: a memory win, not a compute win, on RDNA1; first derivatives, frozen
  base, dims ≤4096, no model-quality claim). Packed INT4 remains hardware-blocked
  (no MFMA on RDNA1); TorchAO/bitsandbytes are not installed.
- Qualified training policies are **FP32/BF16** (per D26–D31); FP16 PEFT and
  broader recipes remain separate qualification work.

## Performance

- **No general two-board speedup claim.** Two-board FSDP can be far slower than
  single-board when the model fits; the L01 application measured only ~1.23×
  warm-update throughput in its particular case.
- Measured wall times are admission costs, not performance guarantees. See
  [PERFORMANCE.md](PERFORMANCE.md) for the bounded owned-workload benchmark.

## Release boundaries

- The release is **immutable**; Torch remains an editable install from the
  materialized tree.
- Rebuilt binaries are **reconstruction proof**, not the shipped artifact: the
  release ships the qualified deployed cohort (some rebuilt daily/rccl binaries
  differ from the deployed cohort).
- Triton provenance is **pin + patches + qualified wheel**; no Triton rebuild
  demonstration (an accepted scope decision).
- **No long-run stability soak** — bounded stability scope only; unknown
  Python/KFD/clocksource causes remain separate.
- Qualification uses **owned fixtures only**; user applications are excluded.

## Deployment / rollback

- Boot fallbacks and rollback assets are preserved on both boards (historical
  cohorts + retained base + boot-health package).
- **Publication/distribution requires explicit instruction.**