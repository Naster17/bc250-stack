# CHECKPOINT — BC-250 gfx1013 full bring-up (all sessions)

> For future agents: this file is the session memory. Start here, then
> `docs/USER-GUIDE.md` for daily use, `docs/TORCH-MATRIX.md` for test
> coverage, `docs/BASELINE-R2.md` for reference numbers.
> Last verified: 2026-09-18. Both boards on r2 default, r1.4 deployed.

## 1. Access (credentials)

- Boards: `nik@192.168.1.105` (main build board), `nik@192.168.1.104`
  (peer). SSH password: `1704`. Same password for `doas` (needs PTY —
  use the `bdoas.py`/`bdoas104.py` PTY wrappers; plain ssh hangs on the
  second `doas` in one session).
- After every reboot: `mountpoint -q /sys/fs/cgroup || doas mount -t
  cgroup2 none /sys/fs/cgroup`, or podman fails.
- GitHub: `gh` authed as `Naster17` (HTTPS push works, SSH too).
- Never: `git push` to stock boot entries, `sched_policy=2`,
  `gpu_recovery` other than 0, manual GPU resets, r1-as-default (r1 was
  one-shot; r2 promoted only after full gates).

## 2. Hardware

- 2x ASRock BC-250 (`1002:13fe`, `gfx1013:xnack-`, salvaged PS5-class APU).
- 40 CUs unlocked (`bc250_cc_write_mode=3`, `simd_count=80`); stock hides
  16. 8 cores / 16 threads online (SMU mask `0xFF`, warm-reboot
  enumerated; health sweep 0 failures incl. unlocked cores 3+7).
- Shared-memory APU: mclk pinned 450 MHz (expected), sclk 15 MHz idle →
  1000 MHz matrix → 2000 MHz sustained GEMM. Power 40–43 W idle,
  120–139 W sustained SGEMM, clean return. `gpu_busy_percent` reads
  "not supported" — use power + clocks, not utilization.

## 3. Software state (both boards)

- Kernel `7.1.5-bc250rocm-r2`, `GRUB_DEFAULT=bc250-rocm-r2-kernel`,
  params `3/0/3/0/0` (cc_write/flush_pasid_kiq/flush_by_runlist/
  sched_policy/gpu_recovery), SDMA navi12 ucode inside r2 initramfs only.
- Runtime image `localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.4`
  (8.35 GB): r1.1 payload + native gfx1013 stack under `/opt/bc250/*`
  + `/opt/torch-venv` (base) + `/opt/torch-venv-dist` (Gloo) +
  `ld.so.conf.d/bc250-r12.conf` + libomp+smi preloads +
  `/opt/bc250/miopen-db` tuned find-DB.
- Torch 2.9.1a0: base wheel `abe93cd8` (133 MB), dist wheel `2d1af013`
  (136.7 MB). `cuda` True, 1 dev, `['gfx1013']`, `numpy<2`, LLVM libomp
  preload, `HSA_ENABLE_SDMA=1`, `OMP_NUM_THREADS=4`.
- Model: `~/models/gemma-4-E4B-it-UD-Q8_K_XL.gguf` (8.1 GB, PPL 63.9747).
  MNIST data: `~/bc250work/mnist/`.

## 4. What was built (in order)

1. **r1 kernel** (6.18.50): 40-CU unlock (duggasco CC+SPI+RLC, PCI-gated),
   PASID KIQ bypass, TTM null guard, KFD runlist TLB rebuild + callsites.
   SDMA hang (>16 KiB HIP-copy) fixed via navi12 ucode in initramfs.
2. **Native prefixes** (GPU-free containers): rocBLAS/hipBLAS (+solver
   rebuild for `getrsBatched`), rocFFT/hipFFT, rocRAND/hiprand,
   rocSPARSE/hipSPARSE, rocSOLVER/hipSOLVER, rocPRIM/hipCUB/rocThrust,
   MIOpen HIP-only (6 one-line gfx1013 gates), hipblaslt + roctracer as
   upstream-deb headers+libs, LLVM libomp staged.
3. **PyTorch 2.9.1** (`v2.9.1` + patches 0004–0012 in
   `bc250-rocm-lab/patches/`): Python hint forwarding, roctx prefix
   fallback, hipblaslt tolerance, roctracer includes, dynamo/functorch
   LANGUAGE C, ABI sync. Wheel via `build-wheel-gfx1013.sh`.
4. **r1.2 image** + validation (GEMM, 50-step train == CPU, llama gate).
5. **r2 kernel** (vanilla 7.1 + 7.1.5 patch, Alpine config seed,
   X86_DECODER_SELFTEST off): all 5 patches port clean; default boot.
6. **Ports**: rocALUTION (CG gate), RPP (add_scalar gate; ROI must be
   device-visible with xnack-), hipDNN SDK (handle gate; vendored deps),
   MIOpen tuning (fwd+bwd Finds, user DB). CK/WMMA/hipSPARSELt/hipTensor
   hardware-blocked (no matrix cores on RDNA1) — documented, not attempted.
7. **Dist torch** (Gloo/TCP): stale-`build/` cache lesson, rocm-smi
   headers+glibc lib needed for symm_mem. Gloo allreduce OK.
8. **r1.3/r1.4 images**, profile promotions, stable containers.
9. **.104 peer**: r2 one-shot → MATRIX_PASS incl FFT (closed KNOWN-FAIL),
   default promoted, r1.4 transferred (sha-verified), torch True.
10. **Review fixes**: atomic apply.sh, roctracer search fix,
    COMPILE_OPTIONS, python-version pin, read-only runlist param, MIOpen
    fp32 provenance note — all pushed + re-verified identical matrix.
11. **Releases** in every fork (see §6).

## 5. Key numbers

- SGEMM 862/3027/4901/5821/5897 GFLOPS (256→4096); torch GEMM identical.
- Conv 0.03–0.9 ms; MIOpen Winograd up to 8x over GEMM fallback in data.
- 50-step train bit-identical GPU/CPU (1.029231/1.004733/0.985372).
- MNIST 0.9865 (5 ep, ~36 s); soak 600 steps 1.0476→0.0073, clean.
- Full matrix 14/14 PASS both boards (see `docs/TORCH-MATRIX.md`).

## 6. Forks / branches / releases

| Repo | Branch (default) | 2026-09 state |
|---|---|---|
| `Naster17/bc250-linux` | `bc250-r2` (r1 legacy) | kernels r1+r2 release `bc250-r1-r2-r1.4` |
| `Naster17/bc250-rocm` | `bc250-gfx1013` | MIOpen + port log, release `bc250-gfx1013-r1.4` |
| `Naster17/bc250-torch` | `bc250-gfx1013` | build fixes + dist notes, release `bc250-gfx1013-dist` |
| `Naster17/llama.cpp` | `gfx1013-rdna1` (on master-default fork) | 1-line fix, release `gfx1013-rdna1-stable` |
| `Naster17/bc250-stack` | `main` | everything, releases `r1.4`, `r1.4-matrix`, `distro-r1` |

Local clones: `~/Documents/bc250-stock/` (review copies) and
`~/bc250-stack/` (persisted), plus board `~/bc250-stack/`
(`bc250-r1`, `bc250-gfx1013` x2, `main`). Profile repo
`~/bc250work/bc250-rocm-profile` (`fe60ef1`).

## 7. Known limits (honest)

- hipblaslt/roctracer: headers+lib only, no gfx1013 kernels. CK/WMMA/
  hipSPARSELt/hipTensor: impossible on RDNA1. No NCCL/flash-attn.
- MIOpen shipped-DB + CK warnings are benign (generic kernels, tuned DB
  wired). PPL corpus unfetchable (HF auth) — reference from r1 stands.
- 3x3 bwd Finds succeed but MIOpen only persisted 1x1 bwd entries —
  training correct, some shapes re-tune per run.
- 2-node Gloo training prepared, never run live (needs both boards +
  MASTER_ADDR recipe in USER-GUIDE §4).

## 8. Resume pointers

- Daily use: `docs/USER-GUIDE.md`. Tests: `docs/TORCH-MATRIX.md`.
- Build scripts: `pytorch/`, `rocm/`, `images/` in this repo (mirrors of
  board `~/bc250work/bc250-rocm-lab/`). Kernel: `Naster17/bc250-linux`
  (`kernel-patches/` + `scripts/apply.sh` here via bc250-linux clone).
- Board todo if returning: PPL with local corpus, 2-node Gloo run,
  long soak on 7.1.5, torch upgrade (needs newer ROCm first).
