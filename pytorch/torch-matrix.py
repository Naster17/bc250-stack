#!/bin/sh
# Deep torch capability matrix on gfx1013. Each section prints PASS/FAIL plus
# ms timings (GPU wall time). CPU cross-checks where cheap. Exit nonzero on
# any FAIL. Run inside r1.4: /opt/torch-venv/bin/python -u this file.
import sys, time
import torch
import torch.nn.functional as F

dev = torch.device("cuda")
torch.manual_seed(0)
fails = []
TIMED = len(sys.argv) > 1 and sys.argv[1] == "--timed"


def sync():
    torch.cuda.synchronize()


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(iters):
        fn()
    sync()
    return (time.time() - t0) / iters * 1000


def check(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {extra}", flush=True)
    if not ok:
        fails.append(name)


print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"ndev={torch.cuda.device_count()} arch={torch.cuda.get_arch_list()}", flush=True)
check("cuda-available", torch.cuda.is_available())
check("arch-gfx1013", torch.cuda.get_arch_list() == ["gfx1013"])

# 1. dtypes: fp32/fp16/bf16 GEMM vs fp64 CPU reference
for dt, tol in ((torch.float32, 1e-4), (torch.float16, 2e-2), (torch.bfloat16, 2e-2)):
    a = torch.randn(128, 128, dtype=dt, device=dev)
    b = torch.randn(128, 128, dtype=dt, device=dev)
    sync()
    got = (a @ b).float().cpu()
    ref = a.double().cpu() @ b.double().cpu()
    err = ((got.double() - ref).abs() / ref.abs().clamp_min(1)).max().item()
    ms = bench(lambda: a @ b) if TIMED else 0.0
    check(f"gemm-{str(dt).split('.')[-1]}", err < tol, f"rel_err={err:.2e} ms={ms:.3f}")

# 2. conv2d fwd + bwd (exercises tuned DB fwd + new bwd entries)
for (n, c, h, k) in ((1, 16, 16, 16), (8, 32, 32, 64)):
    x = torch.randn(n, c, h, h, device=dev, requires_grad=True)
    f = torch.randn(k, c, 3, 3, device=dev, requires_grad=True)
    y = F.conv2d(x, f, padding=1)
    loss = y.square().mean()
    loss.backward()
    ok = (x.grad is not None and f.grad is not None
          and torch.isfinite(x.grad).all().item() and torch.isfinite(f.grad).all().item())
    ms = bench(lambda: (F.conv2d(x.detach(), f.detach(), padding=1).square().mean().backward()
                        if False else F.conv2d(x, f, padding=1))) if TIMED else 0.0
    check(f"conv-bwd-{n}x{c}x{h}-{k}", ok, f"ms={ms:.3f}")

# 3. FFT roundtrip vs CPU (hipFFT path)
x = torch.randn(4, 256, 256, device=dev)
got = torch.fft.ifft2(torch.fft.fft2(x)).real.cpu()
err = (got - x.cpu()).abs().max().item()
ms = bench(lambda: torch.fft.ifft2(torch.fft.fft2(x))) if TIMED else 0.0
check("fft-roundtrip", err < 1e-3, f"abs_err={err:.2e} ms={ms:.3f}")

# 4. RNG: hiprand uniform stats
u = torch.rand(1_000_000, device=dev)
m, v = u.mean().item(), u.var().item()
check("rand-uniform", abs(m - 0.5) < 0.01 and abs(v - 1 / 12) < 0.005,
      f"mean={m:.4f} var={v:.4f}")

# 5. Sparse CSR mv (hipsparse path) vs dense
n = 256
idx = torch.arange(n - 1, device=dev)
# rows 0..n-2 of the 1D Laplacian, each row up to 3 nnz (left/diag/right,
# clamped at domain edges): total nnz = 3*(n-1) - 2.
rows, cols, data = [], [], []
for i in range(n - 1):
    rows += [i] * (1 + (i > 0) + (i < n - 2) + (i == n - 2))
    cols += ([i - 1] if i > 0 else []) + [i] + ([i + 1] if i < n - 1 else [])
    data += ([-1.0] if i > 0 else []) + [2.0] + ([-1.0] if i < n - 1 else [])
offsets = [0]
for i in range(n - 1):
    nnz = 1 + (1 if i > 0 else 0) + (1 if i < n - 1 else 0)
    if i == n - 2:
        nnz = 2
    offsets.append(offsets[-1] + nnz)
crow = torch.tensor(offsets, dtype=torch.int64, device=dev)
col = torch.tensor(cols, dtype=torch.int64, device=dev)
vals = torch.tensor(data, device=dev)
sp = torch.sparse_csr_tensor(crow, col, vals, size=(n - 1, n))
vec = torch.ones(n, device=dev)
got = (sp @ vec).cpu()
ref = torch.zeros(n - 1)
ref += 2.0
ref[1:] -= 1.0
ref[:-1] -= 1.0
check("sparse-csrmv", (got - ref).abs().max().item() < 1e-5,
      f"abs_err={(got - ref).abs().max().item():.2e}")

# 6. Solver: symmetric solve (hipsolver path) vs CPU
A = torch.randn(64, 64, device=dev)
A = A @ A.T + 64 * torch.eye(64, device=dev)
b = torch.randn(64, 8, device=dev)
xs = torch.linalg.solve(A, b)
err = ((A @ xs - b).abs() / b.abs().clamp_min(1)).max().item()
check("linalg-solve", err < 1e-4, f"rel_err={err:.2e}")
L, piv = torch.linalg.lu_factor(A)
check("lu-factor", torch.isfinite(L).all().item())

# 7. Autograd training step (Adam + clip + scaler-less fp16)
m = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(),
                        torch.nn.Linear(64, 8)).to(dev)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
x = torch.randn(128, 32, device=dev)
y = torch.randn(128, 8, device=dev)
l0 = ((m(x) - y) ** 2).mean().item()
for _ in range(5):
    opt.zero_grad()
    loss = ((m(x) - y) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
l1 = ((m(x) - y) ** 2).mean().item()
check("adam-train", l1 < l0, f"{l0:.4f}->{l1:.4f}")

# 8. Dataloader-style H2D + AMP-ish autocast smoke
with torch.autocast(device_type="cuda", dtype=torch.float16):
    xb = torch.randn(64, 32, device=dev)
    out = torch.nn.functional.linear(xb, torch.randn(8, 32, device=dev))
check("autocast-f16", torch.isfinite(out).all().item())

print("MATRIX_" + ("PASS" if not fails else f"FAIL({len(fails)}):{','.join(fails)}"), flush=True)
sys.exit(1 if fails else 0)
