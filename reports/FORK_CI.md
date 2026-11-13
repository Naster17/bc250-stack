# Fork CI — patch-apply reconstruction proof

Added 2026-11-13; **all 8 patched components covered** after the patch-series
reconciliation completed the same day. The public forks carry the BC-250 patches
applied as maintained `bc250/gfx1013` branches; this CI verifies that the
**published patch series in this umbrella actually reconstructs those forks**
from the recorded upstream bases.

## What it checks

`.github/workflows/patch-apply.yml` runs `tools/verify_patch_series.py` per component
(matrix). For each entry in [`ci/components.json`](../ci/components.json) it:

1. fetches only the recorded upstream **base** commit (shallow, no full history);
2. applies the umbrella `patches/<component>/*.patch` series;
3. asserts the resulting **tree hash equals `expected_tree`** — the published fork
   `bc250/gfx1013` tree.

A match proves the series applies cleanly to the upstream base *and* reproduces the
published fork content. It is **git-only** (no ROCm toolchain/build), so it runs on
ordinary GitHub-hosted runners. A full ROCm/Torch build is intentionally out of
scope — that is reproduced from source via the offline recipes on the build host
(see [`../docs/BUILDING.md`](../docs/BUILDING.md)). A weekly schedule surfaces
upstream base drift; a `syntax` job `py_compile`s `tools/`.

### Apply modes

- **`am` (default):** checkout the base, `git am` the series, read `HEAD^{tree}`.
  Used for the five small bases.
- **`index` (manifest `"apply": "index"`):** no checkout at all. Fetch the base with
  `--filter=blob:none` (trees/commits only), `git read-tree` it, `git apply --cached`
  each patch (lazy-fetching only the patched files' preimage blobs), then
  `git write-tree --missing-ok` with `GIT_NO_LAZY_FETCH=1` (plain `write-tree` would
  otherwise lazy-fetch the *entire* missing base blob set). The resulting tree hash
  is identical to what `git am` produces — `tools/tests/verify_patch_series_test.py`
  proves both modes agree on a fixture, plus mismatch/corruption refusals.
  `GIT_LFS_SKIP_SMUDGE=1` is set for every git call: LFS pointer files are the
  tracked content, and no LFS blob is ever needed for tree identity.

The index mode exists because **rocm-libraries' base tree is ~9.4 GiB of git blobs
(54,787 files)** — far too large to check out on a runner (and it is LFS-heavy).
With `blob:none` the base fetch is ~2 MiB and only the ~113 patched files' blobs
are transferred. The BC-250 delta itself touches no LFS-tracked file and contains
no binary diffs.

## Verified components (all PASS, locally reproduced 2026-11-13)

| Component | Upstream base | Patches | Mode | Fork tree |
| --- | --- | --- | --- | --- |
| triton | `66b3f9f` (triton-lang/triton) | 4 | am | `5f62ee9e` |
| migraphx | `9567291` (ROCm/AMDMIGraphX) | 5 | am | `08a36a32` |
| onnxruntime | `402979a` (microsoft/onnxruntime) | 1 | am | `699764ab` |
| torchaudio | `34c52a6` (pytorch/audio) | 4 | am | `a6410cf3` |
| torchcodec | `d9dc542` (meta-pytorch/torchcodec) | 1 | am | `3b481416` |
| rocm-libraries | `3d8f334` (ROCm/rocm-libraries) | 38 | index | `046ed8e3` |
| rocm-systems | `d5c5280` (ROCm/rocm-systems, `hip-version_10.2.62820`) | 8 | index | `65deb63f` |
| pytorch | `c779020` (ROCm/pytorch release/2.13) | 22 | index | `33a86273` |

`expected_tree` pins the published fork content; if a fork is re-published, refresh
the manifest entry (base, published_head, expected_tree, patch_count) and re-verify.

## Reconciliation record (2026-11-13)

The initial CI run (5 components, run 43421509586) exposed three series that did
**not** reconstruct their forks. All three are now fixed, regenerated from the
authoritative local source histories with `git format-patch base..head`:

- **rocm-libraries** — the published series had 33 of the fork's **38** commits;
  the five missing ones were the B07.12-era MIOpen CK-plugin commits (FP32 DL
  plugin facade, descriptor/runtime contracts, FP16 DL gates, FP32 accumulation
  repair, binary-cache versioning). Regenerated as the complete 38-patch series on
  base `3d8f334b9a72ad99a0ecd87552613970433e0500` (ROCm/rocm-libraries develop).
  Verified in index mode (the base worktree checkout that previously hit resource
  limits is no longer needed).
- **rocm-systems** — the published series had 6 of the fork's **8** commits (the
  two RCCL fixes — NVTX includes when tracing is disabled, iostream for DDA IPC
  diagnostics — were missing) and carried duplicate `0001-` numbering. Regenerated
  as the complete, correctly ordered 8-patch series on base
  `d5c528019d0fcfbbc32d96b53ac2c61df6d0e67d` (tag `hip-version_10.2.62820`).
- **pytorch** — `patches/bc250-pytorch` is the historical *incremental* export
  (10 patches on shifting intermediate bases; its README marks it incomplete).
  It is **retained unchanged as historical evidence**. The new
  [`patches/bc250-pytorch-complete`](../patches/bc250-pytorch-complete/README.md)
  directory carries the complete single-base **22-patch** series on
  `c779020750d7dc99a39ea8c104ef8df1fc9d0755` (ROCm/pytorch release/2.13), head
  `8d8b18df558cb478727ba2179f2e01dfce20ea40`. The reconstructed tree
  `33a862732a52c8825a3d1f9c0b609d06fba0df65` exactly matches the D46 verified
  complete-export record (`kernel/artifacts/source-d46/torch-complete.json` on the
  private workstation). CI uses the complete series only.

No fork branch changed — the forks already carried the full histories; only the
umbrella's published series were incomplete. No board access; publication-only.

## Extending

Add a component to `ci/components.json` (upstream, base, patches, fork, fork_branch,
published_head, expected_tree, patch_count, optional apply), verify locally with
`python3 tools/verify_patch_series.py --component <name>`, then add it to the workflow
matrix. Use a `file://` upstream override in a scratch manifest to test against local
source repos without network access (the local repo needs
`uploadpack.allowFilter=true` for index mode).
