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

## Evidence model

Each milestone carries its own receipts with per-file SHA-256. The full internal
evidence archive (build logs, host audit trails, recovery stores) is retained
privately; this public repo carries the curated, scrubbed results and the
reproduction fixtures. The SBOM and source recipes in [`../sbom/`](../sbom/) are
the authoritative reconstruction record.