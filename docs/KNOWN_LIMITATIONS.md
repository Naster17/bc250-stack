# Known limitations

Explicit boundaries of the BC-250 gfx1013 port and the qualified release. These
are not bugs to hide — they are the honest scope. The machine-readable form is
[`../sbom/known-limitations.json`](../sbom/known-limitations.json).

## Hardware

- **No MFMA/WMMA matrix instructions** — cannot be added in software; low-precision
  GEMM/dot paths use scalar/DPP lowering.
- **SDMA disabled**; experimental VM heap disabled (safe defaults).
- **~16 GB VRAM per board**, shared CPU/GPU memory. A single all-resident 1B FSDP
  state floor exceeds physical RAM.
- Advanced hardware profiling (PC sampling/RDC) unqualified; attained-clock /
  temperature / power telemetry semantics unvalidated. `gpu_busy_percent` returns
  EOPNOTSUPP, so direct GPU utilization is unavailable.

## Software subsets

- Scalar backends have documented API/type/operator subsets.
- **Model admission is per-architecture, bounded evidence** — not a universal
  support claim. Admitted: causal LMs (GPT2 / Qwen3 / Qwen3.5-hybrid) plus owned
  pure-torch UNet2D (diffusion-style noise-prediction) and RNN/LSTM/GRU, each
  through load → forward vs CPU/FP64 → backward → optimizer → bitwise resume.
  diffusers is absent (the UNet is owned pure-torch, not a full diffusion
  pipeline); ViT/ConvNeXt, real diffusion pipelines and larger geometries remain
  separate future admissions.
- Production nonsymmetric eig uses host LAPACK (retained; the native small-Schur
  prototype is slower for small orders). **Native ILU(0)/IC(0) sparse
  factorization + preconditioned solve exists** (opt-in, machine precision);
  rocSPARSE has no general direct sparse LU/Cholesky, and extreme
  conditioning/larger orders remain separate from the bounded dense-LU fallback.
- **Math SDPA qualified**; the opt-in tiled/hybrid attention is
  performance-qualified: causal forward 1.5–2.9× faster, full attention 12–86×
  less peak memory but slower compute; dropout/compile boundaries remain
  documented refusals for the tiled path.
- **torch.compile coverage** extends to attention/LoRA/embed+CE/dynamic shapes
  (fullgraph, real Inductor, parity vs CPU + eager) — correctness coverage, not a
  throughput claim (tiny fixtures do not speed up). **HIP graph capture of a
  training step is qualified bitwise-exact** (opt-in): fwd+bwd+eager AdamW,
  whole-step capturable AdamW, checkpointing, fresh-process recapture/resume;
  ~1.9× on a launch-bound stack. Boundaries: static shapes, dropout=0, graph pool
  not checkpointable (resume needs recapture); TRL fused linear-CE boolean
  indexing is not graph-safe. Recorded precision properties (not compile
  defects): FP16 attention backward and dynamic FP16 fused-attention lowering
  differ from eager/CPU at the ~% level.
- **W8 and native NF4/QLoRA quantization exist** (NF4 is explicit opt-in: a memory
  win, not a compute win, on RDNA1; first derivatives, frozen base, dims ≤4096, no
  model-quality claim). Packed INT4 remains hardware-blocked (no MFMA on RDNA1);
  TorchAO/bitsandbytes are not installed.
- Qualified training policies include **FP32/BF16 and FP16 PEFT** (FP16 autocast
  working + FP32 master/frozen/reduce + GradScaler, bitwise resume), including
  distributed FSDP FP16 PEFT with native activation checkpointing and exact
  restart. Broader recipes and larger models remain separate future work.

## Performance

- **No general two-board speedup claim.** Two-board FSDP can be far slower than
  single-board when the model fits; two boards buy capacity and redundancy, not a
  linear speedup.
- The per-operation speed frontier is exhausted: the GEMM term is structural and
  the native fused non-GEMM levers are near-optimal. Further step-time reduction
  needs a model/algorithm-level approach, not per-kernel tuning.
- Measured wall times are admission costs, not performance guarantees. See
  [PERFORMANCE.md](PERFORMANCE.md) for the bounded owned-workload benchmark.

## Release boundaries

- The release is **immutable**; Torch remains an editable install from the
  materialized tree.
- Rebuilt binaries are **reconstruction proof**, not the shipped artifact: the
  release ships the qualified deployed cohort.
- Triton provenance is **pin + patches + qualified wheel**; no Triton rebuild
  demonstration (an accepted scope decision).
- Stability is qualified by a bounded multi-segment soak (exact resume across
  chained segments at stock clocks); indefinite long-run behavior and unknown
  Python/KFD/clocksource causes remain separate.
- Qualification uses **owned fixtures only**; user applications are excluded.

## Deployment / rollback

- Boot fallbacks and rollback assets are preserved (historical cohorts + retained
  base + boot-health package).
- Run the board at **stock clocks** (≤2000 MHz); a persistent SMU overclock
  destabilizes the GPU under sustained load.
- **Publication/distribution requires explicit instruction.**