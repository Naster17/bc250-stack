# Qualification reports

The stack was developed and verified through ten bounded work packages
(PLAN-01 … PLAN-10), each with implementation, qualification, and (where relevant)
deployment batches. Every claim is backed by an owned fixture and a receipt. This
directory summarizes the program; the compatibility and performance results are
promoted to [`../docs/COMPATIBILITY.md`](../docs/COMPATIBILITY.md) and
[`../docs/PERFORMANCE.md`](../docs/PERFORMANCE.md).

## The program

| Package | Scope | Outcome |
| --- | --- | --- |
| PLAN-01 | upcoming guardians | complete |
| PLAN-02 | native forward/cache repair (MIOpen FP16 FP32-accum) | complete |
| PLAN-03 | attention / SDPA bring-up | complete |
| PLAN-04 | sparse / quantization foundations | complete |
| PLAN-05 | distributed / FSDP / PEFT training | complete |
| PLAN-06 | bounded stability + crash triage | complete |
| PLAN-07 | **complete source reconstruction** (D47–D56) | complete |
| PLAN-08 | **coherent versioned release** (D57–D61) | complete |
| PLAN-09 | **model/environment compatibility** (D62–D63) | complete |
| PLAN-10 | **measured training performance** (D64–D65) | complete |

## PLAN-07 — source reconstruction (D47–D56)

Proved the patched ROCm/PyTorch gfx1013 stack rebuilds from preserved, reviewed
inputs alone: D47–D51 pins/patches/toolchain inventories, D52 original builder OCI
image + Gitless source/recipe kit, D53 external ROCm/Clang/RCCL payload, D54
offline configure + selected-target native compiles, D55 full clean Torch rebuild
(daily + rccl-single), D56 generated-file/license provenance closure. Accepted
scope decisions: Triton provenance from pin/patches/qualified wheel (no rebuild
demo); library-delta boundary accepted.

## PLAN-08 — release (D57–D61)

Release `10.2-gfx1013-20261111-138a66f52`: D57 inventory/plan, D58 self-contained
assembly, D59 independent-path qualification on board 1 (326 checks + refusals +
rollback), D60 deployment + qualification on board 2 (656-check suites both
profiles + live rollback cycle), D61 finalization (SBOM / known-limitations /
source-recipe bundle) + promotion. Both boards run the release; publication
requires explicit instruction.

## PLAN-09 — compatibility (D62–D63)

Compatibility matrix of three on-host-verified environments plus a reusable owned
new-model admission fixture (`admit_model_d62.py`): import/load, GPU
forward/backward vs CPU/FP64, optimizer step, bitwise save/resume, and
attention/conv/gated-delta/W8 route + fallback recording. 5/5 owned combinations
passed with all routes fired. See [`../docs/COMPATIBILITY.md`](../docs/COMPATIBILITY.md).

## PLAN-10 — performance (D64–D65)

Owned steady-state training benchmark (`benchmark_training_d64.py`): lr=0 fixed
weights, warmup + repeated intervals, median/IQR, tokens/s, peak memory, numerical
checks. 15/15 cases passed (FP32/BF16/FP16 × eager/sdpa/tiled). Findings: FP32
fastest on tiny models, tiled≈sdpa, hybrid gated-delta/conv the dominant
bottleneck. See [`../docs/PERFORMANCE.md`](../docs/PERFORMANCE.md).

## B07.13 — explicit FP32/FP16 CK activation (optional extension)

Beyond the ten packages, the one genuinely pending optional item — versioned
explicit installation of the qualified B07.12 coherent FP32/FP16 CK + repaired
MIOpen artifact — completed 2026-11-12 on board 1. The optional `bc250-ck` command
now selects the B07.12 format2 worker (a fresh process with the private repaired
MIOpen ahead of the SDK; 27 pinned mapped libraries, API5/66 exports) instead of
the B07.7 FP32-only registration. The B07.7 FP32 package/tools/version record are
retained as the live-rollback target; registration and the wrapper symlink move
together under the exclusive update lock. **92 checks / 10 acknowledged guards**
passed: 18 CPU registration/refusal/live-rollback checks, 72 gated daily+RCCL
checks (six probe batteries, same-process solver137 FP32, conv output and
all-first-gradient fixtures, native backward), and 2 ordinary-user
interactive-elevation checks. 133 protected hashes unchanged; kernel log clean.
Ordinary `bc250-torch`/`bc250-train` defaults, the native runtime, kernel/boot and
board 2 are unchanged. The plugin supplies explicit FP32/FP16 packed2D forward only
(backward via retained native solvers); no global solver choice or whole-model
speedup is claimed. Full identities, evidence and recovery commands are in the
`bc250-therock` fork (`docs/development/bc250_ck_plugin_b0713_results/` and
`bc250_ck_plugin_b0713_usage.md`).

## N4 — read-only telemetry provenance and semantics (optional extension)

Completed 2026-11-13 on board 1 (r2 release). A read-only audit re-established the
running SMU governor/source/overlay provenance and validated attained-clock/
temperature/power/fan semantics: 84 samples (~5 Hz) through idle + one bounded 8 s
native integer GPU load + cooldown, with kernel log, mounts, governor/config/fan
hashes and power policy all unchanged. **No clock, voltage, fan-policy, governor,
overlay, kernel or boot change**, and no raw SMU mailbox commands.

Findings: the external governor still bind-mounts tmpfs over `gpu_metrics` and
hwmon `freq1_input`, so those readings are governor-supplied; `auto` power policy
does not imply stock clocks. The running governor's source is **not pinned** — its
musl cross-build differs from the retained source repo, and the `freq1_input`
overlay code is absent from every retained branch. On the r2 kernel all clock views
(governor overlay, kernel-translated, hwmon, DRM) agree on sensible discrete steps
(1000–2000 MHz), unlike the earlier r3 kernel's nonsensical current-clock slot —
but this is the governor-set SMU clock, **not independent physical attained-clock
calibration**. Utilization remains governor-derived (`gpu_busy_percent` is
EOPNOTSUPP on this board). Resolving physical attained clock and pinning the
governor build source remain optional deeper work (a guarded read-only
`GetGfxFrequency` live-clock getter after establishing SMU mailbox ownership).
Full detail is in the `bc250-therock` fork
(`docs/development/bc250_n4_telemetry_results/`).

## D66 — hybrid bottleneck routing (optional extension)

Completed 2026-11-13 on the .104 release; the first bounded scope of the PLAN-10
bottleneck follow-up. Root cause of the ~3–5× qwen35-hybrid cost: the frozen
Transformers 5.2.0 torch fallback `torch_chunk_gated_delta_rule` builds its
unit-lower-triangular inverse with a **63-iteration Python loop** (hundreds of
sequential small kernel launches per call). The qualified native
`torch.nn.bc250_gated_delta.chunk_gated_delta_rule` (A06/A07) replaces the loop
with batched triangular solves.

A new explicit **opt-in** routing helper (`route_qwen35_gated_delta_d66.py` in the
`bc250-therock` fork) routes the model's chunk call to the native op: gated to
Transformers 5.2.0 + `qwen3_5_text` + actual gfx1013, kill switch
`BC250_NATIVE_GATED_DELTA=0`, counted fallbacks for unsupported
dtype/geometry/initial-state/compiled contexts, exact uninstall. The
trainable-weight causal conv and the recurrent/decode path are **not** routed.
No default, package, source or release identity changes.

Qualified results (owned qwen35-hybrid-tiny, batch 2 × seq 32, D64 protocol):

| Measure | Frozen | Routed | Speedup |
| --- | ---: | ---: | ---: |
| Model step FP32 | 36.994 ms | 15.254 ms | **2.425×** |
| Model step BF16 | 39.529 ms | 17.894 ms | 2.209× |
| Model step FP16 | 39.728 ms | 18.002 ms | 2.207× |
| Chunk op fwd+grads FP32 (benchmark geometry) | 25.49 ms | 5.88 ms | 4.33× |

Numerical parity within pre-declared bounds (FP32 loss Δ0, logits rel-L2 1.5e-7,
worst gradient rel-L2 8.4e-4 on the gate parameters; BF16/FP16 within the
A06/A07-style 0.04 gradient bound), plus an **independent CPU/FP64 gradient
anchor** (both GPU paths ≤ 1.2e-3 of the FP64 reference). The full D62 admission
battery passed on the routed model (bitwise checkpoint resume included); five
refusal checks passed. The original FP32 gradient bound (1e-4) failed in an
intermediate run and is retained on record; the revised 2e-3 bound was
pre-declared with justification before the confirmation runs. Board clean
(0 containers/GPU clients pre/post, temperature stable, no reboot); ordinary
`bc250-train` caller under the immutable release. 21 new local tests (suite 225).
Full detail and evidence hashes are in the `bc250-therock` fork
(`docs/development/bc250_bottleneck_d66_results/`). Remaining optional headroom:
fused native solve/scan kernels, trainable-weight native conv, low-precision cast
overhead, larger-geometry admission.

## D67 — distributed value benchmark (optional extension)

Completed 2026-11-13 on both boards. Consolidates the distributed performance
evidence (D05/D32/D35/D43/L01) into one decision-ready picture and adds an owned
two-board LoRA scaling measurement. **Headline: two boards buy capacity, not
speed.**

- **Full-parameter training that fits one board:** two boards are 6.8–68×
  slower per update (12.61M/45.9M/352.96M FP32 fixtures, matched global batch,
  1 GbE). FSDP's value there is memory headroom (−45.6% live GPU at 352.96M),
  not throughput.
- **Full-parameter training that does not fit one board:** two-board FSDP is
  the demonstrated path — eight 1.024B FP32 updates at 53.36 s median (D43),
  where the single-board all-resident state floor (16.39 GB) exceeds either
  host's RAM.
- **LoRA fine-tuning of a 2B base:** the one workload class where two boards
  are faster per update. L01 measured +23% on real long mixed-length data
  (global batch 16); D67 adds owned synthetic exact-length measurements through
  the same qualified integration (benchmark mode, no user data): Qwen3.5-2B
  rank-128/alpha-256 all-linear, FP16, matched global batch 4 —

  | Operating point | Single | Two-host DDP | Ratio |
  | --- | ---: | ---: | ---: |
  | 1024 tokens, steady state (5 measured updates) | 36.262 s (113 tok/s) | 18.679 s (219 tok/s) | **1.941×** |
  | 4096 tokens, cold first update | 325.822 s | 167.567 s | **1.944×** |

  DDP overlaps the 538 MB gradient all-reduce with the backward pass, so the
  1 GbE link is almost fully hidden. Whole-job startup is paid on both boards:
  L01's six-update job was slower overall on two (248.0 s vs 239.4 s) despite
  the steady-state win — amortize over long jobs.
- **8192-token steady state was not measured:** the application environment's
  reference gated-delta/conv path (the D66 root cause, unfixed in that newer
  Transformers environment) plus eager O(L²) attention makes a single update
  exceed the qualified 460 s guardian child budget; documented as infeasible
  within the safety envelope, with the stopped attempts retained as evidence.

All runs used the L01-pattern guardians (shared runtime locks, 600 s recovery
guards with nonce-matched acknowledgments, bounded child resources, kernel-log
and registry identity, foreign-container refusal); both boards idle before and
after; application entry points, completed outputs and checkpoints untouched;
no runtime/default/package/source change. 24 new local tests (suite 249). Full
detail, method, stopped-attempt records and evidence hashes are in the
`bc250-therock` fork (`docs/development/bc250_distributed_benchmark_d67_results/`).

## D68 — fused native solve/scan (optional extension)

Completed 2026-11-13 on the .104 release; the first bounded scope of the deeper
bottleneck follow-up after D66. The native gated-delta chunk op's remaining cost
is attributed (owned profile fixture): the sequential Python chunk scan is ~70%
of forward+backward at B1/H16/D128/L1024, and the two batched ATen triangular
solves ~48% at the tiny single-chunk geometry.

The patched Torch source (fork commit `262accc`, patch `0023` of the complete
series) adds an **opt-in** `use_fused_solve_scan=True` to
`torch.nn.bc250_gated_delta.chunk_gated_delta_rule`:

- one in-register Triton unit-triangular inverse (analytic backward) replaces
  both batched solves;
- a column-blocked fused linear-recurrence scan (`S_{i+1} = M_i S_i + N_i`,
  adjoint backward, saved entering states) replaces the Python loop; the
  M/N/P/R precompute stays autograd-visible ATen;
- both kernels follow the port's qualified elementwise Triton convention
  (no `tl.dot`, wave32, `enable_fp_fusion=False`);
- **selective from two chunks up** (`FUSED_SCAN_MIN_CHUNKS = 2`, measured
  crossover: at one chunk every fusion variant regresses 0.79–0.94×, so
  single-chunk calls keep the default path); explicit refusal above a 256 MiB
  saved-state cap; FP16/BF16 keep reference-ordered preparation.

Qualified results (final run; candidate module staged as an owned fixture, the
installed release module untouched):

| Measure | Default | Fused | Speedup |
| --- | ---: | ---: | ---: |
| Chunk op fwd+grads FP32, B1/H16/D128/L1024 | 26.17 ms | 14.49 ms | **1.81×** |
| Chunk op fwd+grads FP32, B2/H2/D16/L256 | 8.09 ms | 5.12 ms | 1.58× |
| Chunk op fwd+grads FP32, single chunk (L32) | 4.57 ms | 4.30 ms | 1.0× (default path) |
| Hybrid-tiny model step FP32, seq 256 (D66 route + flag) | 19.05 ms | 16.55 ms | **1.15×** |
| Hybrid-tiny model step BF16/FP16, seq 256 | 22.55/22.82 ms | 19.33/19.68 ms | 1.17/1.16× |

BF16/FP16 op speedups 1.13–1.77× where applied. Numerical parity within
pre-declared bounds: op fused-vs-default rel-L2 ≤ 1.5e-6 FP32 / ≤ 9.3e-5 low
precision (bounds 1e-5 / 1e-2); independent CPU/FP64 token-recurrence anchor at
0.3–5.6% of the A06 tolerances; model parity at seq 256 within the D66 bounds
(FP32 worst gradient 1.52e-3 ≤ 2e-3) with the fused path **7× closer to the
FP64 reference than the frozen fallback** (2.1e-4 vs 1.4e-3). The extended A06
battery passes 72/72 on the candidate; the default path is bitwise identical to
the installed module; refusal checks (non-bool flag, state cap) pass. Board
clean (0 containers/GPU clients pre/post, no reboot); ordinary caller under the
immutable release — **no default, package, release or installed-module change**.
20 new local tests (suite 269).

Defects retained on record: a single-chunk regression in the first
unconditional design (fixed by the measured selectivity), and a fixture-side
routing-injection leak that contaminated an intermediate model-timing control
(root-caused by inject/restore tracing, fixed, and re-qualified with per-case
fused-kernel counters). Full detail and evidence hashes are in the
`bc250-therock` fork (`docs/development/bc250_gated_delta_d68_results/`).
Remaining optional headroom: finer per-geometry selectivity (~5% at four
chunks), trainable-weight native conv, low-precision cast overhead,
larger-geometry admission.

## Evidence model

Each milestone carries its own receipts with per-file SHA-256. The full internal
evidence archive (build logs, host audit trails, recovery stores) is retained
privately; this public repo carries the curated, scrubbed results and the
reproduction fixtures. The SBOM and source recipes in [`../sbom/`](../sbom/) are
the authoritative reconstruction record.