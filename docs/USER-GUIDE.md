# BC-250 Torch + ROCm user guide (both boards)

Both boards now: kernel `7.1.5-bc250rocm-r2` (default boot), runtime image
`localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.4`, 8 cores / 16 threads,
40 CUs. This guide is the only thing you need for daily work.

## 0. First time after any reboot (both boards)

```sh
mountpoint -q /sys/fs/cgroup || doas mount -t cgroup2 none /sys/fs/cgroup
```

Without this, podman fails with "no cgroup mount found". Everything
below assumes it is mounted.

## 1. Run any Python script on the GPU (main pattern)

On **.105** (full lab + launchers present):

```sh
doas ~/bc250work/bc250-rocm-profile/userspace/run-torch.sh your_script.py
```

That launcher sets everything: `HSA_ENABLE_SDMA=1`, `OMP_NUM_THREADS=4`,
Tensile lib path, tuned MIOpen DB, and uses `/opt/torch-venv` inside r1.4.
Your script just does `import torch` — `torch.cuda.is_available()` is True,
arch `['gfx1013']`.

On **.104** (no lab dir, image only), same effect with one podman line:

```sh
doas podman run --rm --device /dev/kfd --device /dev/dri/renderD128 \
  --security-opt seccomp=unconfined --network none \
  -v /home/nik/myproj:/work:ro \
  --env HSA_ENABLE_SDMA=1 --env OMP_NUM_THREADS=4 \
  --env MIOPEN_USER_DB_PATH=/opt/bc250/miopen-db \
  localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.4 \
  /opt/torch-venv/bin/python -u /work/your_script.py
```

Put your project under `/home/nik/...`, mount it `:ro` (or `rw` for
checkpoints), and run. Stdout streams; no logs to hunt.

## 2. Train a model (what works, what to avoid)

Write normal PyTorch. Validated on this stack: Linear/Conv2d/BatchNorm,
Dropout, ReLU, Adam/AdamW/SGD, CrossEntropy/MSE, fp32/fp16/bf16 GEMM,
FFT, RNG, sparse CSRmv, `linalg.solve`, fp16 autocast (see
`pytorch/torch-matrix.py` for the proven list).

```python
import torch
dev = torch.device("cuda")
model = torch.nn.Sequential(
    torch.nn.Linear(32, 64), torch.nn.ReLU(),
    torch.nn.Linear(64, 8)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(600):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

Rules:

- `numpy<2` in your venv (torch built against 1.x).
- Batch sizes that fit 8 GB-ish shared memory; start 128–512.
- fp16/bf16 work but verify numerics (tolerances in matrix doc).
- No NCCL multi-GPU, no flash-attn, no CK fused paths — single GPU,
  math-backend SDPA. Fine to 2k context; longer → llama.cpp.
- MIOpen prints two warnings (missing shipped perf DB, missing CK lib).
  They are harmless: generic kernels, correct results. The tuned user DB
  is already wired via `MIOPEN_USER_DB_PATH`.

## 3. Ready-made: MNIST (.105)

```sh
EPOCHS=100 doas ~/bc250work/bc250-rocm-lab/run-mnist.sh
```

r1.4 + tuned DB + EPOCHS from env (default 5). Expect ~0.98 acc after
2 epochs. Dataset stays in `~/bc250work/mnist/`.

## 4. Two-board training (.105 + .104 over 1 Gbps)

Use the dist venv (Gloo, TCP). Same script both sides, different rank:

```sh
# .105 (rank 0)
MASTER_ADDR=192.168.1.105 MASTER_PORT=29517 WORLD_SIZE=2 RANK=0 \
  doas ... /opt/torch-venv-dist/bin/python -u train_ddp.py
# .104 (rank 1)
MASTER_ADDR=192.168.1.105 MASTER_PORT=29517 WORLD_SIZE=2 RANK=1 \
  doas ... /opt/torch-venv-dist/bin/python -u train_ddp.py
```

with `train_ddp.py` calling
`torch.distributed.init_process_group("gloo")` + `DistributedDataParallel`.
Keep per-rank batches large (Gloo sync over 1 Gbps is the bottleneck;
gradient accumulation helps). Single-proc smoke verified
(`gloo-allreduce: 16.0`); true 2-node run is untested — start with the
MNIST net before your real model.

## 5. llama.cpp inference (.105)

```sh
doas ~/bc250work/bc250-rocm-profile/userspace/run-llama.sh \
  /home/nik/models/gemma-4-E4B-it-UD-Q8_K_XL.gguf --n-predict 64 ...
```

Unchanged. PPL reference 63.9747 (wikitext-2, ctx 2048).

## 6. When something breaks

- `no cgroup mount` → Section 0.
- `cuda.is_available()` False → check `uname -r` is r2 and
  `/dev/kfd` exists; never boot stock for compute.
- Hang with `GPU recovery disabled` in dmesg → power cycle; box returns
  on r2 default on both boards now.
- New project red flags: importing `torch.distributed` + NCCL backend
  (use gloo), `flash_attn` imports, `numpy>=2` in venv.
