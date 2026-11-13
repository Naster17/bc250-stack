# Fork CI — patch-apply reconstruction proof

Added 2026-11-13. The public forks carry the BC-250 patches applied as maintained
`bc250/gfx1013` branches; this CI verifies that the **published patch series in this
umbrella actually reconstructs those forks** from the recorded upstream bases.

## What it checks

`.github/workflows/patch-apply.yml` runs `tools/verify_patch_series.py` per component
(matrix). For each entry in [`ci/components.json`](../ci/components.json) it:

1. fetches only the recorded upstream **base** commit (shallow, no full history);
2. applies the umbrella `patches/<component>/*.patch` series with `git am`;
3. asserts the resulting **tree hash equals `expected_tree`** — the published fork
   `bc250/gfx1013` tree.

A match proves the series applies cleanly to the upstream base *and* reproduces the
published fork content. It is **git-only** (no ROCm toolchain/build), so it runs on
ordinary GitHub-hosted runners. A full ROCm/Torch build is intentionally out of
scope — that is reproduced from source via the offline recipes on the build host
(see [`../docs/BUILDING.md`](../docs/BUILDING.md)). A weekly schedule surfaces
upstream base drift; a `syntax` job `py_compile`s `tools/`.

## Verified components (all PASS, locally reproduced 2026-11-13)

| Component | Upstream base | Patches | Fork tree |
| --- | --- | --- | --- |
| triton | `66b3f9f` (triton-lang/triton) | 4 | `5f62ee9e` |
| migraphx | `9567291` (ROCm/AMDMIGraphX) | 5 | `08a36a32` |
| onnxruntime | `402979a` (microsoft/onnxruntime) | 1 | `699764ab` |
| torchaudio | `34c52a6` (pytorch/audio) | 4 | `a6410cf3` |
| torchcodec | `d9dc542` (meta-pytorch/torchcodec) | 1 | `3b481416` |

`expected_tree` pins the published fork content; if a fork is re-published, refresh
the manifest entry (base, published_head, expected_tree) and re-verify.

## Not yet covered — reconciliation needed (finding)

The same reconstruction test was run against the remaining components and they do
**not** currently pass, so they are excluded from the matrix pending reconciliation:

- **rocm-systems** — applying the 6 published patches to the upstream base yields a
  tree that does **not** match the fork `bc250/gfx1013` head: the fork carries more
  commits than the published series. The series must be regenerated/completed to
  reproduce the fork before it can be CI-verified.
- **rocm-libraries** — the fork carries more commits than the 33 published patches,
  and the repository is LFS-heavy; a local worktree checkout of the base failed on
  resource limits. Needs LFS-aware handling and base/commit reconciliation.
- **pytorch** — the umbrella `patches/bc250-pytorch` directory is the historical
  *incremental* series (its README marks it incomplete and points to a separate
  complete export); it is not a single-base reconstruction series.

These are recorded as follow-up work, not silently dropped: the CI matrix deliberately
lists only the components whose series provably reconstruct the published fork.

## Extending

Add a component to `ci/components.json` (upstream, base, patches, fork, fork_branch,
published_head, expected_tree, patch_count), verify locally with
`python3 tools/verify_patch_series.py --component <name>`, then add it to the workflow
matrix. Use a `file://` upstream override in a scratch manifest to test against local
source repos without network access.