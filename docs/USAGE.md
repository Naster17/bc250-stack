# BC-250 stack — final usage guide

**What this is, what is qualified, and how / what / with-what to run it.**
Updated 2026-11-16. For deep reference see [ARCHITECTURE.md](ARCHITECTURE.md),
[BUILDING.md](BUILDING.md), [COMPATIBILITY.md](COMPATIBILITY.md) and
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

---

## 1. The hardware reality (read first)

The BC-250 APU exposes an AMD **`gfx1013`** GPU — **RDNA1 (Navi)**, not CDNA.
Consequences that no software refresh can change:

- **No MFMA/WMMA matrix cores.** GEMMs run on the vector ALUs (Tensile FP16
  kernels); throughput is bounded by that, not by a matrix engine.
- **wave32**, FP16 rapid-packed-math, **FP64 at 1/16 rate**.
- **~16 GB VRAM** per board, shared with the host CPU allocation.
- Upstream ROCm does **not** officially support `gfx1013`; this stack is the port
  that makes it work.

Two boards buy **capacity and redundancy, not a single 32 GB device and not a
linear speedup** (measured in D67: LoRA ~1.94× per update across two boards;
full-parameter training that fits one board is slower distributed). Per-operation
speed levers were exhausted by D71/D78/D79 (the GEMM term is structural; the
native fused non-GEMM levers are near-optimal). Further step-time reduction needs
a model/algorithm-level approach, not per-kernel tuning.

## 2. What is installed and qualified

| Item | Value |
| --- | --- |
| Deployed release (both boards) | `10.2-gfx1013-20261111-138a66f52` |
| Runtime profiles | `daily` (default, NCCL-disabled) and `rccl-single` (RCCL-enabled, separately qualified) |
| Kernel | r3 `7.2.10-bc250rocm-r3` on both boards; retained r2 boot fallback |
| Installed Torch native code | `a524980` cohort (daily + rccl-single) |
| Torch published source | D80 refresh `b0c641e` on `ROCm/pytorch release/2.13` tip `02aa33a` (RC tag `v10.2-gfx1013-torch-refresh-20261116-rc1`); 23-patch series |
| ROCm sources | Libraries `e1d8d1ff`, Systems `143f1d878` (qualified 10.2 BKC line) |
| Hosts | .104 (`bc250`, development/build) + .105 (`bc500`, deployment) |
| Public repos | GitHub `Naster17/bc250-stack` umbrella + 10 component forks (`bc250/gfx1013`) |

The D80 refresh is **parity-preserving** (bit-identical forward losses, exact
bitwise resume, all five native routes fire) and is published as **source**; it
does not change the deployed binary release. Refreshing the ROCm monorepo
(libraries/systems) is a separate full re-port, kept on the qualified 10.2 BKC
line unless a concrete need arises.

## 3. Single-host usage (run on a board as `nik`)

The launchers select the installed patched Torch/ROCm foundation in a bounded
container. Your home, working directory and script parent are mounted at their
original absolute paths; files saved there survive the container exit.

| Command | Use |
| --- | --- |
| `bc250-torch` | Python with the patched Torch/ROCm foundation; bring your own package environment. |
| `bc250-train` | Same foundation + the frozen training consumers (Transformers/PEFT/Accelerate). |
| `bc250-migraphx` | Registered MIGraphX inference SDK (application must select its GPU target). |
| `bc250-ort` | Registered ONNX Runtime / MIGraphX (explicit GPU execution provider). |
| `bc250-ck` | ABI-pinned MIOpen scalar DL FP32 forward option (.104); preserves plain defaults. |

```bash
bc250-torch --launcher-help                 # inspect launcher options
bc250-torch /absolute/path/app.py           # run your app
bc250-train --runtime-profile daily train.py
bc250-train --runtime-profile rccl-single train.py   # RCCL-enabled cohort
```

- The default profile is `daily`. `--runtime-profile rccl-single` selects the
  matched RCCL-enabled Torch cohort; it is **not** a general speed switch and its
  single-device qualification does not imply multi-board support.
- `bc250-train` mounts `/opt/bc250-stack/current/python-consumers` read-only and
  loads its `training.env`; use it for the frozen consumers, `bc250-torch` for an
  independently managed environment (do not unintentionally mix package sets).
- External dataset/model disks need an explicit `--mount` / `--mount-ro`; a symlink
  needs its **target** mounted. See [BUILDING.md](BUILDING.md).

## 4. Two-host distributed training (.104 + .105)

One native `gfx1013` GPU process per board; master `.104`, transport `eth0`
sockets, backend **`nccl`** (AMD **RCCL**). Both boards must be idle; each needs
its own local copy of the same code/config/data shard. The launcher does not copy
code or start the remote peer.

```sh
bc250-distributed --info        # inspect the provider on each node
# On .104:
bc250-distributed --node-rank 0 --master-port 29861 /absolute/path/train.py
# On .105 (same code/settings, same free port, start promptly):
bc250-distributed --node-rank 1 --master-port 29861 /absolute/path/train.py
```

A training script consumes `RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT`
and initializes its process group with `env://`.

**Qualified scope (D01–D77):**

| Mode | Qualified |
| --- | --- |
| DDP | FP32 transformer gradients/AdamW/checkpoint references; BF16/FP16 residual MLP + scaler; equal-batch accumulation/no_sync; public consumer env/mounts. |
| Classic FSDP `FULL_SHARD` | Nested FP32 + 12.61M transformer/AdamW parameter/gradient/optimizer shards; BF16/FP16 accumulation, local-shard overflow/scaler replay; same-world checkpoint replay. |
| FP16 PEFT FSDP (D74) | FP16 working / FP32 master+reduce, native world-group `ShardedGradScaler`, frozen-aware gradient gate; exact restart. |
| FP16 PEFT + activation checkpointing (D77) | D74 precision + native non-reentrant RNG-preserving checkpointing; exact restart; both blocks recompute. |
| RCCL collectives | FP32/FP64/FP16/BF16 sums across both physical hosts. |

**Re-verified 2026-11-16** on the current release: the D77 headline path
(`frozen` seed → fresh-process `frozen` resume with exact resumed transcript +
different birth, FP32 reductions, `ShardedGradScaler`) and the retained D74
checkpointing-off control all **passed on both ranks**, boot-guards acknowledged,
kernel log + distributed registry unchanged.

**Not qualified:** large-model memory peaks, full production convergence,
training speedup, multi-GPU-per-host, P2P/IB, other protocols. Application-level
two-host DDP entry points can be built on this launcher with their own package
sets; they need not use the frozen D23 FSDP policy.

## 5. Native opt-in capabilities (PLAN2 + D66–D80)

All are **opt-in**; plain Torch/train defaults are bitwise unchanged. The pytorch
patch series stays at 23 patches.

| Capability | How | Qualified result |
| --- | --- | --- |
| NF4 / QLoRA (PLAN2-02) | native dequantize-to-FP16 helper | persistent weight memory 0.281× FP16 (~3.56×); peak 0.916×; QLoRA loss 0.0056→0.0019. Native INT4 is hardware-blocked on RDNA1. |
| Sparse factorization (PLAN2-03) | native ILU(0)/IC(0) hipSPARSE + csrsv2 preconditioned solve + RHS gradient | machine-precision; nonsymmetric eig retains host-LAPACK. |
| Tiled SDPA attention (PLAN2-04) | `attn_implementation="bc250_tiled"` (zero dropout) | causal fwd 1.5–2.9× faster; full attention 12–86× less peak memory (slower compute). |
| Compile / HIP graph capture (PLAN2-05, D76) | `GraphedStep` helper | bitwise-exact capture incl. capturable AdamW + checkpointing; ~1.9× on the owned launch-bound stack; real LoRA step ~1.0014× (GEMM-bound). TRL fused linear-CE boolean indexing is **not** graph-safe. |
| FP16 PEFT (PLAN2-07) | FP16 autocast + FP32 master/frozen/reduce + GradScaler | 12/12 checks, bitwise resume. |
| Hybrid gated-delta route (D66/D68/D69) | opt-in native fused solve/scan | 2.2–2.4× step on the frozen hybrid path; fused delta op 1.13–1.81×. |
| Native causal conv (D73) | route Transformers 5.22.1 `causal_conv1d_fn` | 80/80 parity; LoRA step 1.0145×/1.0187×, combined with fused delta 1.0283×. |
| Broader model admission (PLAN2-06, D75) | D62-style battery | UNet2D/RNN/LSTM/GRU + ~0.12B Qwen3.5-hybrid admitted; bitwise resume. |

See [COMPATIBILITY.md](COMPATIBILITY.md) and [reports/](../reports/) for the
qualified scope and evidence.

## 6. Redistributable Docker image

A self-contained runtime image (ROCm `gfx1013` libraries + patched Torch +
training consumers) is **published on Docker Hub** so users can run the stack
**without building from source**. It is assembled FROM the qualified release
base, baking the release ROCm prefix + Torch + consumers at the launcher's
absolute paths (runtime parity preserved). GPU-verified on `gfx1013`.

> **Public image:** `docker.io/naster17/bc250-stack:10.2-gfx1013-20261111`
> (4.38 GB compressed / ~17.3 GB uncompressed). GPU probe passed
> (`torch 2.13.0a0+gfx1013`, `cuda True`, `gfx1013:xnack-`, device matmul). The
> ROCm prefix is split into per-subdirectory layers to stay under Docker Hub's
> 10 GB per-layer limit. See [BUILDING.md](BUILDING.md) and [REPOS.md](REPOS.md).

Run pattern:

```bash
docker pull naster17/bc250-stack:10.2-gfx1013-20261111
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video -v "$PWD":/work -w /work \
  naster17/bc250-stack:10.2-gfx1013-20261111 python3 app.py
```

The image targets `gfx1013` only; it does not enable other AMD architectures.

## 7. Support matrix and honest limitations

- **Supported:** native `gfx1013` (RDNA1) single-host training/inference with the
  patched Torch/ROCm; two-host DDP/classic-FSDP within the qualified scope above;
  the opt-in native capabilities in §5.
- **Not supported / not qualified:** other AMD architectures (no MFMA path),
  multi-GPU-per-host, P2P/IB/SDMA, general distributed protocols beyond the
  qualified two-host socket layout, large-model convergence/speedup claims,
  native INT4 (hardware-blocked), graph capture of TRL fused linear-CE.
- **Memory:** ~16 GB VRAM per board shared with host CPU allocations; two boards
  do not expose one 32 GB device.
- **Clocks:** boards run at stock clocks (≤2000 MHz, cyan-skillfish governor +
  OD cap). A persistent SMU overclock caused host-level reboots and is removed;
  never modify the host governor/clock/watchdog/kernel unasked.

## 8. Where things live

| Location | Meaning |
| --- | --- |
| GitHub `Naster17/bc250-stack` | Umbrella: docs, patch series, recipes, SBOM/licenses, reports. |
| GitHub `Naster17/bc250-*` forks | Patched component sources on `bc250/gfx1013` branches. |
| Board `/opt/bc250-stack/current` | Installed runtime selection (versioned release cohort). |
| Container `/workspace/pytorch-work` | Patched Torch tree (read-only). |
| Container `/workspace/build/dist/rocm` | Native ROCm prefix (read-only). |

Reproducibility: the published patch series reconstructs each fork tree (CI
patch-apply proof); the offline recipes pin exact source commits, builder image,
dependency closure and resource envelope. See [BUILDING.md](BUILDING.md) and
[REPOS.md](REPOS.md).