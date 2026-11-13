# BC-250 repository map

The stack is published as one umbrella repository (`bc250-stack`, this repo) plus
a set of component forks. Every fork carries a maintained **`bc250/gfx1013`**
branch with the BC-250 patches applied directly to the code, based on a recorded
upstream pin so the branch stays rebasable against upstream. Unmodified
dependencies are **recipe-fetched pins**, not forks.

Naming convention: `bc250-<component>`. Branch convention: `bc250/gfx1013`.

## Component forks

| Repo | Upstream | Base pin | Patches | Role |
| --- | --- | --- | --- | --- |
| `bc250-linux` | Alpine Linux 7.1 (patched-snapshot import) | kernel 7.1.5-bc250rocm-r2 | 6 (r3) | Kernel: amdgpu/amdkfd BC-250 compute quirks |
| `bc250-therock` | `ROCm/TheRock` | our build branch | — | ROCm superbuild / build system |
| `bc250-rocm-libraries` | `ROCm/rocm-libraries` | `3d8f334b9a7` (develop) | 38 | rocBLAS/hipBLASLt/MIOpen/etc.; MIOpen gfx1013 FP16 repair + CK plugin |
| `bc250-rocm-systems` | `ROCm/rocm-systems` | `hip-version_10.2.62820` (`d5c528019d`) | 8 | HIP/ROCr/RCCL/profiler; gfx1013 visibility fixes |
| `bc250-torch` | `ROCm/pytorch` | `c779020750` (release/2.13) | 22 | PyTorch: native gfx1013 training/attention/sparse/quant ops |
| `bc250-triton` | `triton-lang/triton` | recorded merge-base | 4 | Triton: gfx1013 low-precision dot / RDNA1 descriptors |
| `bc250-torchaudio` | `pytorch/audio` | `v2.11.0` | 4 | torchaudio gfx1013 RNNT/IIR fixes |
| `bc250-torchcodec` | `meta-pytorch/torchcodec` | `v0.17.0` | 1 | torchcodec gfx1013 seek fix |
| `bc250-migraphx` | `ROCm/AMDMIGraphX` | `rocm-10.1` (`95672916e`) | 5 | MIGraphX: C++17/20 + rocBLAS-default fixes |
| `bc250-onnxruntime` | `microsoft/onnxruntime` | `v1.31.0` (`402979a`) | 1 | ONNX Runtime MIGraphX EP lifetime fixes |

> **Torch base note:** the patched Torch branch is based on **`ROCm/pytorch`**
> (the ROCm fork), *not* `pytorch/pytorch` mainline. Fork `ROCm/pytorch` so the
> base commits exist and the series applies cleanly. The reconstructible series
> is [`patches/bc250-pytorch-complete`](../patches/bc250-pytorch-complete/README.md)
> (22 patches, single base `c779020750`); `patches/bc250-pytorch` is the retained
> historical incremental export and must not be combined with it.

## Recipe-fetched pins (no fork — unmodified upstream)

| Dependency | Pin | Notes |
| --- | --- | --- |
| torchvision | `v0.28.0` (`8fb87713`) | built pristine for gfx1013; no source patches |
| llvm-project | `ROCm/llvm-project` (recorded) | compiler; unmodified |
| half / hipify / rocm-cmake | recorded | TheRock submodules |
| abseil / protobuf / pybind11 | recorded | MIGraphX third-party deps |

The authoritative per-file pins and licenses are in [`sbom/`](../sbom/) and
[`recipes/`](../recipes/). The full reconstruction proof is PLAN-07 (D47–D56).

## Kernel provenance (important)

`bc250-linux` is **not** a clean upstream fork: the r2 source is an
already-patched snapshot imported from the legacy build (Alpine Linux 7.1 base,
"not a pristine upstream download"), and r3 is a 6-patch delta on top. The repo
preserves the patched source tree, the r3 patch series, the `.config`, and the
provenance/anchor hashes so the exact running kernel is reproducible. See
[`docs/KERNEL.md`](KERNEL.md).

## Umbrella repository contents (`bc250-stack`)

```
bc250-stack/
├── README.md            # overview + quickstart + repo map
├── docs/                # architecture, building, compatibility, performance, kernel
├── patches/             # every patch series, by component (also applied in the forks)
├── recipes/             # offline build recipes (D52 kit + D55 full rebuild)
├── sbom/                # SBOM, license manifest, source recipes, known limitations
├── reports/             # curated qualification reports (release + PLAN-07..10)
└── tools/               # bc250 build/qualification tooling
```

The patch series are kept here **in addition to** being applied in the forks so
they can be reviewed, regenerated (`git format-patch`), and upstreamed
independently of the fork branches.