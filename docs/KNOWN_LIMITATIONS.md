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
- Production nonsymmetric eig uses host LAPACK; scalable sparse factorization
  differs from the bounded dense-LU fallback.
- **Math SDPA qualified**; tiled/hybrid attention has dropout/compile/graph
  boundaries.
- **W8 quantization exists**; NF4 / packed INT4 / full QLoRA are separate
  unqualified feasibility work.
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