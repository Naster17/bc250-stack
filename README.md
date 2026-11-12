# bc250-stack

Native **ROCm + PyTorch** for the **AMD BC-250** APU (Cyan Skillfish, **gfx1013**,
RDNA1) — a maintained, reproducible port that makes the board a real training
device, not just a display adapter.

The BC-250's gfx1013 is a consumer RDNA1 part that upstream ROCm/PyTorch do not
target: it has no MFMA/WMMA matrix cores, SDMA is disabled, and several compute
paths (attention, low-precision dots, sparse, quantization, the kernel compute
queues) need explicit fixes. This stack carries those fixes as a clean, rebasable
set of branches and patch series, with a full reconstruction proof and a
qualified, versioned release.

> **Status:** all ten work packages complete. Release
> `10.2-gfx1013-20261111-138a66f52` is built, qualified, and deployed on both
> boards with proven rollback. See [`docs/`](docs/) and [`reports/`](reports/).

## What's here

This is the **umbrella repository**: architecture and usage docs, every patch
series, offline build recipes, the SBOM/license bundle, and curated qualification
reports. The actual patched source lives in the component forks below, each on a
maintained `bc250/gfx1013` branch.

| Layer | Repo | What it carries |
| --- | --- | --- |
| Kernel | [`bc250-linux`](https://github.com/Naster17/bc250-linux) | amdgpu/amdkfd BC-250 compute quirks |
| Build system | [`bc250-therock`](https://github.com/Naster17/bc250-therock) | ROCm superbuild |
| ROCm libraries | [`bc250-rocm-libraries`](https://github.com/Naster17/bc250-rocm-libraries) | rocBLAS/hipBLASLt/MIOpen gfx1013 repairs |
| ROCm systems | [`bc250-rocm-systems`](https://github.com/Naster17/bc250-rocm-systems) | HIP/ROCr/RCCL/profiler fixes |
| Compiler | [`bc250-triton`](https://github.com/Naster17/bc250-triton) | gfx1013 low-precision dot / RDNA1 descriptors |
| Framework | [`bc250-torch`](https://github.com/Naster17/bc250-torch) | native gfx1013 training/attention/sparse/quant ops |
| Consumers | [`bc250-torchaudio`](https://github.com/Naster17/bc250-torchaudio) · [`bc250-torchcodec`](https://github.com/Naster17/bc250-torchcodec) | audio/codec gfx1013 fixes |
| Graph compilers | [`bc250-migraphx`](https://github.com/Naster17/bc250-migraphx) · [`bc250-onnxruntime`](https://github.com/Naster17/bc250-onnxruntime) | MIGraphX / ONNX Runtime EP fixes |

Full map with upstream pins and patch counts: [`docs/REPOS.md`](docs/REPOS.md).

## Architecture at a glance

```
        ┌──────────────────────────────────────────────────────────┐
        │  Training apps / consumers (torch, torchaudio, torchcodec) │
        ├──────────────────────────────────────────────────────────┤
        │  PyTorch (bc250-torch)  +  Triton (bc250-triton)           │
        │  native gfx1013 ops: attention, gated-delta, sparse, W8    │
        ├──────────────────────────────────────────────────────────┤
        │  ROCm libraries (bc250-rocm-libraries) + systems           │
        │  hipBLASLt / MIOpen / RCCL / HIP / ROCr / profiler         │
        ├──────────────────────────────────────────────────────────┤
        │  Kernel (bc250-linux): amdgpu/amdkfd compute quirks        │
        ├──────────────────────────────────────────────────────────┤
        │  BC-250 APU — gfx1013 (RDNA1, no MFMA/WMMA, SDMA off)      │
        └──────────────────────────────────────────────────────────┘
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the layer-by-layer
explanation, the gfx1013 constraints, and how the pieces fit together.

## Quickstart

- **Use the release:** see [`docs/BUILDING.md`](docs/BUILDING.md) and the release
  notes. The release is a self-contained, build-free extractable prefix.
- **Build from source:** the offline recipes in [`recipes/`](recipes/) reproduce
  the stack from the fork branches + pinned dependencies (the PLAN-07 proof).
- **Study / contribute:** start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
  then the patch series in [`patches/`](patches/).

## Documentation

| Doc | Contents |
| --- | --- |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | layers, gfx1013 constraints, design |
| [REPOS.md](docs/REPOS.md) | repo map, upstream pins, patch counts |
| [BUILDING.md](docs/BUILDING.md) | build-from-source + release usage |
| [KERNEL.md](docs/KERNEL.md) | kernel provenance and r3 patch series |
| [COMPATIBILITY.md](docs/COMPATIBILITY.md) | tested package/kernel-route matrix |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | measured training performance + bottlenecks |
| [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | explicit boundaries |

## Guarantees and boundaries

- **Reproducible:** PLAN-07 (D47–D56) proved a clean offline rebuild from the
  preserved inputs; the SBOM and license bundle are in [`sbom/`](sbom/).
- **Qualified:** the release passed independent-path qualification on both boards
  (326 + 656 checks) with live rollback; see [`reports/`](reports/).
- **Honest scope:** this is a bounded port for gfx1013. It is not a universal
  speedup or model-quality claim; see
  [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

## License

Each component retains its upstream license. The wheel/license bundle is in
[`sbom/license-manifest.json`](sbom/license-manifest.json). Original BC-250 port
contributions are MIT unless a component's upstream license requires otherwise.