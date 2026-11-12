# Building and using the stack

Two paths: **use the qualified release** (recommended), or **build from source**
(the reconstruction path). The offline recipes are in [`../recipes/`](../recipes/).

## Path A — use the release

The release `10.2-gfx1013-20261111-138a66f52` is a **self-contained, immutable,
build-free extractable prefix**. It ships the qualified deployed cohort (ROCm +
materialized Torch trees + consumer wheels) with `daily` and `rccl-single`
profiles. Extract it, point the ordinary `bc250-*` callers at it, and run. No
compiler toolchain is needed to *use* it.

The release layout, SBOM, and source recipes are in [`../sbom/`](../sbom/).

## Path B — build from source (reconstruction)

PLAN-07 (D47–D56) proved the stack rebuilds **offline** from preserved inputs
alone. The recipes reproduce this:

- [`recipes/build-kit/build-fresh-daily.sh`](../recipes/build-kit/build-fresh-daily.sh)
  and `build-fresh-rccl-single.sh` — independent pip-only Torch candidates.
- [`recipes/full_build_d55.sh`](../recipes/full_build_d55.sh) +
  `run_full_build_d55_on_host.py` — the full clean rebuild of both profiles.

### Build envelope (important)

The Torch build is deliberately constrained to match the board:

| Setting | Value | Why |
| --- | --- | --- |
| `PYTORCH_ROCM_ARCH` | `gfx1013` | the only target |
| `MAX_JOBS` | `4` | ~15 GiB RAM binds, not cores; `-O3` HIP TUs take 1–2 GiB each |
| `CMAKE_JOB_POOLS` | `link=1` | serialize the big link spikes |
| `BLAS` | `OpenBLAS` | host BLAS |
| `USE_FLASH_ATTENTION` / `USE_MEM_EFF_ATTENTION` | `0` | no efficient/flash on gfx1013 |
| `USE_ROCM_CK_GEMM` / `USE_ROCM_CK_SDPA` / `USE_MSLK` | `0` | CK/MSLK not for RDNA1 |
| `USE_NCCL` | `0` (daily) | RCCL via the `rccl-single` profile |
| `USE_DISTRIBUTED` / `USE_GLOO` | `1` | Gloo distributed |
| `PIP_NO_INDEX` | `1` | fully offline |
| ccache | read-only, 20 G | reconstruction uses the preserved cache |

Do **not** raise parallelism just because cores are idle — RAM is the binding
constraint; a larger envelope must be a separately measured admission experiment.

### Inputs

The build consumes the fork branches (`bc250-torch`, `bc250-triton`,
`bc250-rocm-libraries`, `bc250-rocm-systems`, …) plus the recipe-fetched pins
(torchvision, llvm-project, etc.) recorded in [`../sbom/source-recipes.json`](../sbom/source-recipes.json).
The external ROCm/Clang/RCCL payload and toolchain are preserved per D53/D51.

## Verifying a build

The qualification fixtures live in [`../tools/`](../tools/). The release was
qualified by independent-path activation on both boards (326 + 656 checks) with
live rollback; the owned admission and benchmark fixtures
(`admit_model_d62.py`, `benchmark_training_d64.py`) reproduce the compatibility
and performance evidence. See [`COMPATIBILITY.md`](COMPATIBILITY.md) and
[`PERFORMANCE.md`](PERFORMANCE.md).