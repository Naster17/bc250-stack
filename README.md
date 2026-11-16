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

> **Status:** the stack is complete, qualified, and released. A redistributable
> Docker image is **published on Docker Hub**:
> [`naster17/bc250-stack:10.2-gfx1013`](https://hub.docker.com/r/naster17/bc250-stack)
> — run the stack without building from source. See
> [`docs/USAGE.md`](docs/USAGE.md) and [`docs/`](docs/).

## What's here

This is the **umbrella repository**: architecture and usage docs, every patch
series, offline build recipes, and the SBOM/license bundle. The actual patched
source lives in the component forks below, each on a maintained `bc250/gfx1013`
branch.

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
        ┌─────────────────────────────────────────────────────────────┐
        │  Training apps / consumers (torch, torchaudio, torchcodec)  │
        ├─────────────────────────────────────────────────────────────┤
        │  PyTorch (bc250-torch)  +  Triton (bc250-triton)            │
        │  native gfx1013 ops: attention, gated-delta, sparse, W8     │
        ├─────────────────────────────────────────────────────────────┤
        │  ROCm libraries (bc250-rocm-libraries) + systems            │
        │  hipBLASLt / MIOpen / RCCL / HIP / ROCr / profiler          │
        ├─────────────────────────────────────────────────────────────┤
        │  Kernel (bc250-linux): amdgpu/amdkfd compute quirks         │
        ├─────────────────────────────────────────────────────────────┤
        │  BC-250 APU - gfx1013 (RDNA1, no MFMA/WMMA, SDMA off)       │
        └─────────────────────────────────────────────────────────────┘
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the layer-by-layer
explanation, the gfx1013 constraints, and how the pieces fit together.

## Run from Docker (no build)

The full stack is a ready-to-run image on Docker Hub — pull it and pass through
the GPU:

```bash
docker pull naster17/bc250-stack:10.2-gfx1013
docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video \
  -v "$PWD":/work -w /work naster17/bc250-stack:10.2-gfx1013 python3 app.py
```

[`naster17/bc250-stack`](https://hub.docker.com/r/naster17/bc250-stack) — tags
`10.2-gfx1013` / `10.2` / `latest`, following the **ROCm release line** (a future
ROCm 10.3 gets its own `gfx1013` tag after re-porting + qualification). Requires a
`gfx1013` (RDNA1) AMD GPU with the `amdgpu` driver. Full requirements + examples:
[`docs/USAGE.md §6`](docs/USAGE.md).

## Quickstart

- **Start here:** [`docs/USAGE.md`](docs/USAGE.md) — what is qualified and how /
  what / with-what to run (single-host, two-host distributed, native opt-in
  capabilities, and the redistributable Docker image).
- **Use the release:** see [`docs/BUILDING.md`](docs/BUILDING.md). The release is
  a self-contained, build-free extractable prefix.
- **Build from source:** the offline recipes in [`recipes/`](recipes/) reproduce
  the stack from the fork branches + pinned dependencies (a clean offline rebuild
  is proven).
- **Study / contribute:** start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
  then the patch series in [`patches/`](patches/).

## Documentation

| Doc | Contents |
| --- | --- |
| [USAGE.md](docs/USAGE.md) | usage guide: what is qualified, how/what/with-what to run |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | layers, gfx1013 constraints, design |
| [REPOS.md](docs/REPOS.md) | repo map, upstream pins, patch counts |
| [BUILDING.md](docs/BUILDING.md) | build-from-source + release usage |
| [KERNEL.md](docs/KERNEL.md) | kernel provenance and patch series |
| [COMPATIBILITY.md](docs/COMPATIBILITY.md) | tested package/kernel-route matrix |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | measured training performance + bottlenecks |
| [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | explicit boundaries |

## Guarantees and boundaries

- **Reproducible:** a clean offline rebuild from preserved inputs is proven; the
  SBOM and license bundle are in [`sbom/`](sbom/).
- **Qualified:** the release passed independent-path qualification with live
  rollback on the target hardware.
- **Honest scope:** this is a bounded port for gfx1013. It is not a universal
  speedup or model-quality claim; see
  [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

## License

Each component retains its upstream license. The wheel/license bundle is in
[`sbom/license-manifest.json`](sbom/license-manifest.json). Original BC-250 port
contributions are MIT unless a component's upstream license requires otherwise.