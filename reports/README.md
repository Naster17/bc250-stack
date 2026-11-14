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

## D69 — route portability to Transformers 5.22.1 (optional extension)

Completed 2026-11-13 on the .104 release, in the application's newer
Transformers 5.22.1 environment (the lora-smoke integration, unchanged).
Reconstruction showed 5.22.1 already replaced the 5.2.0 63-iteration inverse
loop with batched `solve_triangular` and calls the **module-level global**
`torch_chunk_gated_delta_rule`, so the D66 instance patch does not apply; the
remaining launch-bound piece is the Python chunk scan — the same structure as
the installed native op's default path.

New owned tooling in the `bc250-therock` fork (no Torch source change): a
module-scope gated route (version/device/model-type gates, kill switch, a
kwargs classifier that ignores-and-records the control flags the torch
reference itself ignores and falls back — counted — on packed/positioned/
non-causal/unknown inputs, exact-identity uninstall), an explicit training
wrapper around the unchanged app entry point, an app-environment qualification
fixture, and D67-pattern guardians with resume.

**Qualified:** op parity vs the 5.22.1 reference FP32 ≤ 7.3e-7 rel-L2,
BF16/FP16 **bitwise 0**; CPU/FP64 recurrence anchor at 0.2–5.3% of the A06
tolerances; tiny owned Qwen3.5 model parity within the D66 bounds (FP32 worst
gradient 1.25e-3 ≤ 2e-3; native 7× closer to FP64 than the reference); 7
refusal checks. LoRA re-check on D67's hash-verified owned synthetic data
(global batch 4, rank-128 FP16 eager + checkpointing): route records show
864/144 native calls with **zero fallbacks** and exact uninstall.

| Case | Reference | Route → installed | Route → D68 fused |
| --- | ---: | ---: | ---: |
| 1024 steady s/update | 36.166 | 36.189 (1.000×) | 35.670 (**1.014×**) |
| 4096 cold first update s | 324.172 | — | 317.671 (**1.020×**) |
| 8192 first update | infeasible (D67) | — | **still > 460 s envelope** (safely stopped) |

**Honest finding:** the port is clean, but the step-level gain is only ~1–2%.
The chunk op is a small share of the whole LoRA step — eager full attention,
the conv reference, whole-model launch/dispatch overhead (24 layers × 4
microbatches × gradient checkpointing) and the 134M-parameter optimizer
dominate. This refines D67's attribution: "launch-bound gated-delta" holds at
the op level (D68 measured 1.8× there), but the app step is launch-bound
across the whole model. The next lever for long-sequence LoRA is broader
launch reduction (qualified fused/tiled attention, conv, or graph capture) —
separate bounded scopes. No application source/entry point/output/checkpoint
change, no user data, no default/package/release change; boards clean
(guardian kernel-log/registry gates, nonce-acked boot guards, boot unchanged).
16 new local tests (suite 285). Full detail and evidence hashes:
`bc250-therock` fork, `docs/development/bc250_route_port_d69_results/`.

## D70 — LoRA step decomposition + launch-reduction feasibility (optional extension)

Completed 2026-11-13 on the .104 release (app environment, D67-pattern
guardians). First bounded scope of the broader launch-reduction direction:
**measure where the ~36 s LoRA update actually goes and prove whether HIP
graph capture / torch.profiler work on this stack, before building any
attention/conv route.** Measurement only — owned span-instrumentation wrapper
around the unchanged app entry point (transparent CUDA-event spans, class-level
phase patches, an instance-level patch on the prepared optimizer, layer/MLP
hooks, conv/chunk/eager function wraps; finalized post-run, no in-step
synchronization), a 7-section feasibility probe, and D67-pattern guardians
with resume. No Torch source change (pytorch series stays 23 patches).

**Reproduction:** 1024 steady 36.118 s/update (0.999× the D69 reference
36.166 s), final loss identical (2.4717), peak 6.17 GiB; 4096 cold first
update 329.770 s (1.017× D69 324.172 s), peak 6.99 GiB. GPU-timeline coverage
~1.000, CPU wall closure 0.000 s, zero unpaired hooks, exact uninstall.

**Finding 1 — the 1024 step is GEMM-compute-bound, not launch-bound.** The
kernel census (torch.profiler, kernel events separated from their parent
aten:: ops to avoid double-counting) shows **157,675 kernel launches and
36.32 s of device-busy time per update — the GPU is ~100% busy** — and ~89%
of that time is Tensile FP16 GEMM kernels executing the LoRA-adapted
projections at very low efficiency (dominant skinny-tile family
`MT64x1x64…GSU3`: 1176 calls × 14.5 ms = 17.1 s per update). GPU timeline:
linear-attention mixer 54.7%, MLP 32.5%, full-attention mixer 5.8% (the eager
attention *core* is 0.3%), optimizer **0.1%** (30 ms — the D69
"optimizer dominates" hypothesis is refuted), conv+chunk ~3%. Host gaps
overlap GPU execution; launch overhead is already hidden.

**Finding 2 — the ~9× step growth at 4096 is not attention either.** The
linear-attention mixer grows **10.9×** for 4× the tokens (215.1 s, 65.2% of
the step — superlinear LoRA GEMM cost), MLP 7.2×, the conv reference **20×**
(0.44 → 8.95 s); the eager attention core grows 14.8× ≈ L² as theory predicts
but is only 0.5% of the 4096 step. 8192 infeasibility is therefore GEMM/conv
scaling — attention routing alone would not restore it. (Cold-vs-steady
growth factors are indicative attribution, not a controlled scaling study.)

**Finding 3 — HIP graph capture is feasible and bitwise-exact on this
stack.** All seven probe sections passed: torch.profiler CUDA activity;
graph capture of MLP fwd+bwd (12.01 → 5.63 ms, 2.13×), tiny Qwen3.5 eager
ATen (2.09×), the D68 fused **Triton** path (2.47×, 75 native calls, zero
fallbacks inside the captured region), **with gradient checkpointing**
(3.44×), and `make_graphed_callables` — all with bitwise-identical loss and
gradients on replay. On the real model, though, capture can only remove the
~11% non-GEMM share plus already-overlapped host overhead; its value grows
once GEMM time shrinks.

**Re-ranked backlog (evidence-backed):** (1) GEMM efficiency for the LoRA
shapes — qualify the explicit CK FP16 path (`bc250-ck`, B07.13) on these
shapes / Tensile skinny-N solution tuning / LoRA A/B GEMM batching; a 2× GEMM
improvement would cut the 1024 step ~45%; (2) trainable-weight native
depthwise causal conv (20× growth); (3) graph-capture integration; (4) D68
headroom (~1–3%); fused/tiled attention routing is **demoted** (0.3–0.5% of
the step at these lengths). Defects found and fixed while qualifying are
retained in the report (tiny-model SDPA default, stale-graph capture block,
checkpoint hook semantics diagnosed on-stack, accelerate's inner-optimizer
re-patch vs class-level patching, async budget/census double-counting,
single-update progress print). 20 new local tests (suite 305). No application
source/entry point/output/checkpoint change, no user data (D67 owned
synthetic data reused by hash), no default/package/release change; boards
clean every phase (kernel-log/registry gates, nonce-acked boot guards, boot
unchanged, 0 containers). Full detail and evidence hashes: `bc250-therock`
fork, `docs/development/bc250_launch_reduction_d70_results/`.

## D71 — GEMM efficiency for the LoRA shapes (optional extension)

Completed 2026-11-13 on the .104 release (app environment, D67-pattern
guardians). First bounded scope of D70's evidence-ranked #1 lever, reusing
the qualified N32 method (private-handle `rocblas_gemm_ex_get_solutions`
enumeration, CPU-double parity, canary guards, paired alternating timing).
Measurement and qualification only — no preset deployment, no application
source change, no user data.

**Headline: the easy GEMM win does not exist on this stack.**

- **Census:** 27,240 GEMM calls/update across 71 distinct layouts
  (TorchDispatchMode, exact sizes/strides/dtypes).
- **Solution sweep (40 mm/addmm layouts):** every FP16 layout has exactly
  **2 admitted Tensile solutions** (contrast N32's BF16 case: 14/14/8);
  best-vs-default paired ratios 0.96–1.05× — within noise. **Zero preset
  winners.** All candidates pass the unchanged N32 parity contract.
- **hipBLASLt is slower:** fresh-process AB/BA ratios 0.28–0.60× on the top
  shapes; rocBLAS remains the right backend.
- **Whole-step A/B:** reference 36.196 s (1.0008× the D69 reference),
  candidate preset (release rows + 0 winners) 36.158 s (**1.001×**), final
  loss **bit-identical** (Δ=0), finite gradients, no skipped steps, peak
  6.16 GiB. The override mechanism is validated end-to-end but empty for FP16.
- **Per-shape in-model profile** (one microbatch × grad_accum): GEMM
  **32.8 s/update**, all-kernel **36.3 s/update ≈ the step** (GPU ~100% busy,
  ~90% GEMM) — **D70's GEMM-bound attribution independently confirmed.**

**Two corrections to the framing:** (1) the cost is the **large base-model
projections** (`[1024,6144]@[6144,2048]` at 35 ms/call in-model, ~16 s/update),
not the LoRA adapters (rank-128 A/B GEMMs are 0.25 ms/call, ~0.4 s/update);
(2) isolated hot-cache fixtures understate the in-model cost ~4× (35 vs
8.8 ms on the same logical GEMM) — the slowdown is memory/state-dependent
(cold activations, locality, and/or the port's backward `fp16_alt_impl`
flag, which the fixtures and override CSV do not exercise), **not a
bad-solution choice**. Fixture-based solution tuning is not decision-grade
for this workload.

**Revised recommendation:** stop FP16 solution-selection tuning (exhausted).
The tractable levers are now native depthwise causal conv (D70: 20× growth),
graph-capture integration (~10% non-GEMM, D70 proven bitwise-exact), and the
D68 headroom; a GEMM breakthrough requires dedicated microarchitectural R&D
(resolve the in-model-vs-fixture gap, then memory-traffic reduction / fusion).

**Incident and hardening:** a full-update `record_shapes` profile thrashed
the host; the orchestrator's session timeout closed the SSH connection and
SIGHUP-killed the guardian before it wrote the boot-guard ack (ack is written
only in cleanup), and the boot-guard child died with it, so the designed
emergency reboot did not fire; an orphaned container held ~10 GB. The user
power-cycled .104 (r2 kernel and release intact, no data loss, nothing to
roll back; the deployment host was untouched). Hardened: one-microbatch
profile window, a `timeout 430` podman backstop so a container can never
outlive a killed guardian, and a longer orchestrator session; the resource
envelope is unchanged. Defects found and fixed are retained in the report
(census config staging, SystemExit capture, PosixPath JSON, `--mount-ro`
syntax, profiler OOM → stream aggregation, audit schema). 22 new local tests
(suite 327). No application source/entry point/output/checkpoint change, no
user data, no default/package/release change. Full detail and evidence
hashes: `bc250-therock` fork,
`docs/development/bc250_gemm_efficiency_d71_results/`.

## D72 — GEMM in-model-vs-fixture gap diagnostic (optional extension)

Completed 2026-11-13 on the .104 release (D67-pattern guardians, D71-hardened
envelope). D71 left the dominant base projection's ~4× in-model-vs-fixture
gap (35 vs 8.8 ms/call) unexplained and listed three candidate causes without
testing them. D72 tests all three with isolated GEMM microbenchmarks — no
model training, no full-model profile (avoiding the D71 host-thrash vector),
no source/default/release/preset change, no user data.

**Headline: none of the three cheap explanations reproduces the gap; the GEMM
cost is structural/systemic, not a per-GEMM property.**

| Hypothesis | Measurement | Ratio | Verdict |
| --- | --- | ---: | --- |
| Backward `fp16_alt_impl` flag | fwd 9.217 ms vs bwd 9.446 ms; identical solution set `[-318,-319]` for both flags; all 8 swept mm layouts 0.987–1.025× | 1.025× | ruled out |
| Thermal / clock throttling | 21 windows × 221 calls over 36 s flat ~7.8 ms/call; clock pinned 2000 MHz throughout; temp ≤ 77 °C | 1.002× | ruled out |
| Memory locality (cold pool) | hot 7.828 ms vs cold 7.826 ms | 0.9997× | ruled out |
| Memory locality (fresh write) | hot 7.828 ms vs fresh 8.017 ms | 1.024× | ruled out |

The port sets `rocblas_gemm_flags_fp16_alt_impl` (0x4) on backward fp16 GEMMs
(`CUDABlas.cpp`, via `ROCmBackwardPassGuard`); D71's sweep used `flags=0` only
and never exercised that path — D72 shows it changes nothing on gfx1013. The
sustained probe also closes N4's attained-clock question for sustained GEMM
load (read-only hwmon, no SMU mailbox): the clock holds 2000 MHz and throughput
is flat. The dominant layout is taken from D71's per-shape in-model profile by
device time, not a geometry proxy (the two large projections tie on K×N).

**Attribution: `not_reproduced_by_isolated_fixtures`** (pre-declared thresholds
1.5 meaningful / 2.5 dominant, above D71's 0.96–1.05 noise band; every ratio
~1.0). The fixture runs the dominant GEMM at ~7.8–9.4 ms under *every* isolated
condition vs 35 ms in-model — the 3.7–4.5× gap survives all tests. The gap is a
**systemic in-model execution effect**; the remaining untested candidates are
whole-model working-set/TLB pressure (2 B parameters + activations + optimizer
state vs one ~37 MB tensor pair), kernel concurrency/queueing (~157 k
launches/update), and profiler device-time attribution including wait time.
None is addressable by per-GEMM tuning (flag / solution / locality).

**This reinforces D71's bottom line and narrows the GEMM R&D scope:** the
tractable near-term levers remain native depthwise causal conv, graph-capture
integration, and the D68/non-GEMM remainder; a GEMM breakthrough requires deep
microarchitectural investigation of the systemic effect, now with the cheap
causes eliminated.

Defect found and fixed: the first sustained probe queued an unbounded async
backlog whose single sync consumed the whole budget (one window, trivial
degradation); fixed with calibrated bounded synchronized windows + a regression
test, the buggy run retained as labelled evidence. 23 new local tests (suite
350). Board boot unchanged, three guardians passed, kernel log and distributed
registry unchanged, no foreign containers, all boot-guards acknowledged, idle
post-run; the deployment host was untouched. Full detail and evidence hashes:
`bc250-therock` fork, `docs/development/bc250_gemm_gap_d72_results/`.

## PLAN2-01 — production-readiness stability soak (optional extension)

Completed 2026-11-14 on the .104 release (r2 kernel, stock 2000 MHz). First item
of the forward PLAN2 capability-expansion plan. **Result: PASSED** — the stack is
reliable for sustained multi-hour LoRA training with exact checkpoint/resume.

**20 chained bounded segments · 200 optimizer updates · 2.073 h accumulated ·
all 9 reliability checks green.** Global steps contiguous 1→200 (exact resume
across every segment boundary), per-update drift 0.9992, cross-segment memory
ratio 1.000 (no leak), temperature ≤65.85 °C inter-segment (≤90 °C bound), mean
update 37.31 s, every update finite + non-skipped, loss 2.936→2.025 (finite,
decreasing), adapter changed each segment, 20/20 guardians passed, boot ID
unchanged, kernel log + distributed registry clean, no foreign containers; the
deployment host was untouched.

Method: a multi-hour run accumulated as many ≤460 s guardian segments joined by
exact checkpoint/resume (cumulative `max_steps`, 10 updates/segment), which also
validates restart reliability. Authorized lora-smoke app, single-board
Qwen3.5-2B rank-128 FP16 LoRA, eager attention + gradient checkpointing, global
batch 4, owned hash-verified synthetic data; D71-hardened guardians (SH lock,
nonce-acked boot-guard, `timeout 430` backstop) with durable host state for
resume. One pre-declared refinement: the kernel-log identity check ignores only
benign governor `Unsupported clock type` clock-query noise; real GPU faults still
fail (tested). The original qualified guardian is otherwise unchanged.

Incident journey (documented in the evidence archive): the first attempt's
segment 2 spontaneously rebooted the host. Read-only root cause: a persistent
**3500 MHz SMU overclock** (applied at boot) plus **`amdgpu.gpu_recovery=0`** →
GPU hang under load → hardware/firmware reset. Ruled out the boot-guard (deadline
not reached), the SP5100 watchdog (`bootstatus=0`), and kernel panic (no `panic=`).
This is a **host-level configuration issue, not a stack defect**. The user
stabilized the host to stock clocks; benign governor kernel noise was handled by
the pre-declared filter; a non-root GPU monitor (`amdgpu_top`) was closed to free
the device; then the clean 2 h run completed. No host governor/clock/watchdog/
kernel was modified by this work. 27 new local tests (suite 377). No
default/package/release change.

The production-readiness gate is met **conditional on a stable host** (stock
clocks; the aggressive overclock is outside the qualified envelope). Full detail
and evidence hashes: `bc250-therock` fork,
`docs/development/bc250_soak_plan2_01_results/`.

## PLAN2-02 — native NF4 / QLoRA quantization (optional extension)

Completed 2026-11-14 on the .104 release (r2 kernel, Torch 2.13.0a0+gfx1013,
stock ≤2000 MHz). Second item of the forward PLAN2 plan. **Result: PASSED** — a
native, opt-in 4-bit NormalFloat (NF4) weight-only recipe with a qualified
forward/backward, a QLoRA composite, a measured memory reduction and D62
admission. **70/70 GPU qualification checks + D62 routes-only admission passed.**

Feasibility decision (read-only first): bitsandbytes / TorchAO are absent, and the
native packed-INT4 path (`_weight_int4pack_mm` / `_convert_weight_to_int4pack`) is
**hardware-blocked** on gfx1013 — it requires a CDNA2 architecture gate *and* MFMA
matrix instructions that RDNA1 physically lacks (plus BF16-only input). The
admitted route is therefore **native NF4 via dequantize-to-FP16 + the qualified
rocBLAS GEMM**: a pure-Python helper mirroring the proven frozen-W8 pattern, with
**no new native kernel and no Torch source edit** (the pytorch patch series stays
at 23). NF4 is a **memory win, not a compute win**, on RDNA1 (no INT4 matrix
cores).

Recipe: the canonical 16-level NormalFloat-4 codebook, block-wise absmax
quantization (block sizes 16/32/64/128/256, default 64), two 4-bit codes per byte,
FP32 block scales, nearest level via sorted midpoints. A custom autograd function
saves the **compact codes/scales** (no floating base persisted) and recomputes the
dequantized weight for the input first derivative; `once_differentiable`, frozen
base. `FrozenNF4Linear` (strict versioned state, CPU storage/load, gfx1013
forward) and `QLoRALinear` (frozen NF4 base + trainable adapter). An independent
pure-Python scalar reference shares only the codebook constants; this is an
NF4-compatible recipe qualified against a CPU-double reference, **not** a
bitsandbytes bit-exactness claim.

Measured: persistent weight bytes **0.281× FP16 (~3.56× smaller)**; peak allocated
memory **0.916×** on a 24-layer owned fixture (the persistent saving dominates as
model size grows; a tiny fixture's eager dequant transient can mask the peak win).
Pack code-match 1.000, dequant rel-L2 3.8e-08; compute parity max rel-L2 FP16
0.00045 / BF16 0.00353 / FP32 2.9e-07 (within pre-declared bounds). QLoRA 16 BF16
AdamW steps: loss 0.00560→0.00188 (finite, decreasing), adapter changed, base
bitwise frozen. 14 refusals plus the preserved native INT4-conversion /
int8-backward boundaries. D62 admission fired the new `quant-nf4` route alongside
attention / gated-delta / conv / W8.

Attempt journey (test-harness fixes, not stack defects; retained as evidence): an
early code-match check compared indices to packed bytes, and an unfrozen QLoRA
source was correctly refused; then FP16 AdamW `exp_avg_sq` underflow diverged to
NaN (fixed with BF16 adapters, the QLoRA-standard stable choice) and a tiny memory
fixture masked the win (fixed with a memory-lean dequant + larger fixture); then
70/70. No pre-declared bound was relaxed. 34 new local tests (suite 413). Guardian
= the qualified PLAN2-01 envelope plus a stock-clock gate (`pp_dpm_sclk` ceiling
≤2000 MHz). No default/package/release/source change; not installed into any
image; the deployment host was untouched.

Boundaries: first derivatives only, frozen base, operand/work bounds (dims ≤4096;
wider real-model projections need a separate admission); no bitsandbytes/TorchAO
installation, no packed-INT4 enablement, and **no model-quality/perplexity claim**
(the ~9% recipe rel-Frobenius on Gaussian weights is the 4-bit quantization error).
Full detail and evidence hashes: `bc250-therock` fork,
`docs/development/bc250_nf4_plan2_02_results/`.

## PLAN2-03 — native sparse factorization + nonsymmetric-eig disposition (optional extension)

Completed 2026-11-14 on the .104 release (r2 kernel, Torch 2.13.0a0+gfx1013,
stock ≤2000 MHz). Third item of the forward PLAN2 plan. **Result: PASSED** —
native general-CSR ILU(0)/IC(0) sparse factorization with a qualified
preconditioned iterative solve and RHS first gradient, essentially at machine
precision. **42/42 GPU qualification checks.** Nonsymmetric eig retains the
qualified host-LAPACK default (documented decision, no rework).

Feasibility decision (read-only, user-confirmed): **rocSPARSE has no general
*direct* sparse LU/Cholesky** on any architecture. The native options are
ILU(0)/IC(0) preconditioners (`csrilu0`/`csric0`, general CSR, **wave32-aware** so
gfx1013-feasible — unlike the MFMA-blocked packed INT4) and banded direct
`gtsv`/`gpsv`. The admitted subset is **ILU(0)/IC(0) + a bounded preconditioned
solve** — the scalable native alternative to the B02.1 dense-LU fallback (which
densifies and does not scale). The active-release `libhipsparse` exports
`csrilu02`/`csric02`/`csrsv2` + the Info/zero-pivot APIs (confirmed by `nm -D`).
The pinned general Torch sparse-solve API rejects ROCm, so this is a
**private-handle ctypes hipSPARSE path** reusing the N2.13 pattern — **no native
rebuild, no Torch/library source edit.**

Method: strict no-fill ILU(0) (IKJ) / IC(0) with numeric boost disabled and zero
pivot reported (never silently perturbed), against an independent pure-Python CPU
reference (same algorithms, forward/back substitution, dense Gaussian-elimination
oracle, CSR transpose). The ILU factors split into unit-lower L and upper U; the
preconditioner M⁻¹ = U⁻¹L⁻¹ is applied with native `csrsv2` (analysis is
trans-specific; the adjoint uses separate TRANSPOSE-analyzed solvers) inside an
outer Richardson iteration. The RHS first gradient (dL/db = A⁻ᵀ dL/dx) uses the
adjoint preconditioner M⁻ᵀ = L⁻ᵀU⁻ᵀ.

Measured (worst case): ILU(0) factor rel-L2 FP32 1.27e-07 / FP64 9.7e-18; IC(0)
FP32 1.64e-08 / FP64 0.0; `csrsv2` lower/upper ~1e-16; preconditioned-solve
relative residual FP32 8.2e-08 / FP64 1.4e-16; solution rel-L2 FP64 4.2e-16; RHS
gradient FP32 3.9e-08 / FP64 2.9e-16 — all far within pre-declared bounds.
Dense-pattern ILU(0) reconstructs A exactly; zero/singular pivot matches the CPU
reference; 7 refusals pass.

Nonsymmetric-eig disposition (recorded, no implementation): production eig stays
**host-LAPACK** (N2.2). The extensive native small-Schur prototype (N2.3–N2.12)
was measured **slower than host LAPACK for small matrices** (N2.6: host ~0.2–0.4 ms
vs native ~3.75–11.8 ms), and the practical review retains host-LAPACK and
deprioritizes tiny-eig optimization. Larger/ill-conditioned native eig remains a
separate deep scope.

Attempt journey (harness/API fixes, not stack defects; retained): a staging path
bug; classic hipSPARSE naming (`hipsparseCreate`/`Destroy`, not `*Handle`);
factorization 27/27; `csrsv2` trans-specific analysis (TRANSPOSE solve after a
NON_TRANSPOSE analysis → status 3) fixed with TRANSPOSE-analyzed adjoint solvers
plus a corrected singular fixture; then 42/42. No pre-declared bound relaxed. 15
new local tests (suite 428). Guardian = the qualified PLAN2-01/02 envelope plus a
stock-clock gate. No default/package/release/source change; not installed into any
image; the deployment host was untouched.

Boundaries: bounded owned fixtures (orders ≤256, FP32/FP64, well-conditioned);
**no general direct sparse LU/Cholesky claim**; extreme conditioning, hard-spectra
convergence contracts, larger orders, complex dtypes, compile/graph integration
and performance tuning are separate; first derivatives only (RHS gradient); the
solve is iterative with a reported residual, not direct. Full detail and evidence
hashes: `bc250-therock` fork, `docs/development/bc250_sparse_plan2_03_results/`.

## PLAN2-04 — attention speed qualification (optional extension)

Completed 2026-11-14 on the .104 release (r2 kernel, Torch 2.13.0a0+gfx1013,
stock ≤2000 MHz). Fourth item of the forward PLAN2 plan. **Result: PASSED** —
the EXISTING opt-in gfx1013 tiled SDPA (`torch.nn.attention._gfx1013`, three
Triton scalar kernels: online softmax, FP32 accumulation, recomputed backward,
GQA without atomics) is now performance-qualified on long-sequence owned
fixtures. **63/63 qualification checks + 168/168 benchmark checks** (41 paired
math-vs-tiled fixtures, 0 unstable). Scope user-confirmed: qualify + document
only — **no Torch source/default change** (pytorch series stays 23 patches);
math SDPA remains the production default.

Evidence gap closed: the earlier A07 "BF16 tiled is slower" timings were all
head_dim 256 **hybrid** fixtures (math FP32 forward + tiled backward); pure-tiled
FP16/BF16 at head_dim ≤128 — the common LLM geometry — had never been paired
against math at long sequence.

Measured tradeoff (paired A1/B1/B2/A2, 2 warmups + 5 samples, median, HIP
events; parity verified before every timed fixture):

- **Causal attention**: tiled forward **1.5–2.9× faster** (masked-tile skipping);
  fwd+bwd up to ~2.0× at head_dim 64, mixed 0.67–1.4× at head_dim 128 GQA.
- **Full (non-causal) attention**: tiled **slower** (fwd+bwd 0.19–0.69×) but
  **12–86× less peak memory** — e.g. head_dim 64, 16 heads, L=4096: 4152 MiB
  math vs 48.2 MiB tiled; head_dim 128 GQA L=4096: 4208 vs 72.2 MiB. This is the
  long-sequence **capacity** unlock: full attention that would OOM with math
  becomes feasible within 16 GB.
- The recomputed scalar backward is the weak point; the BF16 head_dim>128 hybrid
  keeps the math forward's quadratic temporary (memory ratio only 0.66–0.80).

Parity worst rel-L2 vs an independent CPU-FP64 anchor (output + dQ/dK/dV, 39
cases incl. masks with fully-masked rows, cross-attention, GQA, head256,
L up to 4096): FP32 7.6e-07 / FP16 4.3e-04 / BF16 3.4e-03 (pre-declared bounds
3e-5 / 5e-3 / 3e-2, never relaxed). 13/13 refusals raise the documented
exception (dropout, mask gradients, causal+mask, GQA-without-flag, non-finite
scale, D>256, seq>4096, FP64, dtype mismatch, CPU, wrong rank, higher-order,
compile). Integration: non-reentrant checkpoint parity; tiny Transformers Llama
`bc250_tiled` + checkpointing vs CPU-double math (FP32 worst grad rel-L2
7.2e-07); autocast FP16/BF16; empty dims. Informational probes: CUDA-graph
capture of the tiled forward **works and replays bitwise-exact** on a small
fixture (graph integration/qualification remains PLAN2-05 scope); math
no_grad+GQA works in this release.

Attempt journey (harness fixes, not stack defects; all retained): the release
container lacks transformers (fixed via the owned lora-smoke env site-packages,
soak pattern); the higher-order refusal was first tested at the wrong level
(`once_differentiable` detaches grads on the create_graph call; the RuntimeError
surfaces when differentiating them again); two bench runs crashed at the first
GQA fixture and were initially misdiagnosed as a no_grad+GQA math-backend
boundary — instrumented probes proved the real bug: the timing helpers dropped
`is_causal`/`enable_gqa` from every timed call (so the early D64 "causal"
numbers were non-causal on both backends). Those numbers are retained as defect
evidence and used for no claim; all fixtures were re-measured with opts
forwarded, and a regression test guards the fix. No bound, drift gate or
envelope parameter was relaxed. 23 new local tests (suite 451). Guardian = the
qualified PLAN2-01/02/03 envelope plus a stock-clock gate. No default/package/
release/source change; not installed into any image; the deployment host was
untouched.

Positioning: opt-in for long-seq causal speed/memory or full-attention capacity;
not a default switch and not a LoRA-step speedup (D70: the eager attention core
is 0.3–0.5% of the step). Full detail and evidence hashes: `bc250-therock` fork,
`docs/development/bc250_attention_plan2_04_results/`.

## Evidence model

Each milestone carries its own receipts with per-file SHA-256. The full internal
evidence archive (build logs, host audit trails, recovery stores) is retained
privately; this public repo carries the curated, scrubbed results and the
reproduction fixtures. The SBOM and source recipes in [`../sbom/`](../sbom/) are
the authoritative reconstruction record.