# BC-250 model/environment compatibility matrix

A record of which package combinations and kernel routes are **actually
tested/qualified** on the native gfx1013 stack. This is not a promise list: a
combination appears here only with bounded evidence. Architecture names alone do
not establish support; each model is admitted by the owned admission fixture
(below) plus its recorded route/fallback behavior.

- Native provider: patched Torch/Triton/ROCm for `gfx1013:xnack-`.
- Status key: **qualified** = bounded evidence on the release; **frozen** =
  pinned fixture scope, retained; **absent/unqualified** = not installed or not
  proven.

## 1. Native provider (shared by every environment)

| Component | Identity |
| --- | --- |
| Torch | `2.13.0a0+gfx1013` |
| Triton | gfx1013 low-precision dot / RDNA1 descriptors |
| ROCm | 10.2 release line (libraries + systems), native gfx1013 |
| Runtime profiles | `daily` (NCCL-disabled) and `rccl-single` (RCCL-enabled) |
| Target arch | `gfx1013:xnack-` (no MFMA/WMMA; SDMA off) |

No environment below replaces this provider. Package overlays/venvs link to it;
they do not freeze or substitute the native Torch/ROCm binaries. Exact source
revisions are pinned in the SBOM ([`sbom/`](../sbom/)).

## 2. Qualified Python package environments

Distinct consumer environments are qualified. They are **not interchangeable**:
installing a newer Transformers into one does not qualify the other's helpers or
a new model's kernels.

### 2a. Frozen training consumers (`bc250-train`)

| Package | Version |
| --- | --- |
| transformers | 5.2.0 |
| accelerate | 1.12.0 |
| peft | 0.18.1 |
| torch | 2.13.0a0+gfx1013 (native provider) |
| triton | 3.8.0 (native provider wheel) |
| numpy | 2.5.1 |
| huggingface-hub | 1.4.1 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |

- Qualified scope: owned GPT2 causal-LM (tied head), FP32/BF16/FP16 FSDP policies,
  frozen-base + native/PEFT LoRA adapters, non-reentrant activation checkpointing,
  bitwise checkpoint save/resume. **FP16 PEFT** (frozen base + LoRA, FP16 autocast
  + FP32 master + GradScaler) is qualified with bitwise resume; the distributed
  frozen FSDP PEFT helpers remain FP32/BF16.
- The Trainer helpers **pin** these versions and refuse on mismatch; a source or
  package change requires helper requalification (see §5).

### 2b. Application environment (`bc250-distributed`, newer Transformers)

A separate application environment uses a newer Transformers stack:

| Package | Version |
| --- | --- |
| transformers | 5.22.1 |
| accelerate | 1.15.0 |
| peft | 0.21.2 |
| trl | 1.17.1 |
| datasets | 5.1.0 |
| torch | 2.13.0a0+gfx1013 (native provider, linked) |

- Qualified scope: native DDP two-host LoRA on a Qwen3.5-2B-class hybrid (full base
  per board, FP32 LoRA + AdamW, RCCL over Ethernet), shifted-target loss
  normalization, sealed-checkpoint replay.
- This is an **application** environment, separate from the frozen fixtures. The
  newer application Trainer is not the source-pinned frozen Trainer.
- Not a general Qwen-family or two-board speedup guarantee.

### 2c. Qwen3.5 native-op overlay (separate Transformers 5.19.0)

| Package | Version |
| --- | --- |
| transformers | 5.19.0 |
| torch | 2.13.0a0+gfx1013 (native provider) |

- A **third** Transformers version, distinct from the frozen 5.2.0 consumers and
  the 5.22.1 application stack. The native-op overlay refuses unless
  `transformers.__version__ == "5.19.0"` and `model_type == "qwen3_5_text"`.
- Qualified scope: owned gated-delta + tiled/hybrid attention + frozen causal-conv
  ops on the hybrid Qwen3.5 architecture. Environment toggles restore references.
- The hybrid Qwen3.5 **architecture** is also present in the frozen transformers
  5.2.0 and admits there with the reference (torch) gated-delta/conv path. The
  5.19.0 gate applies to the **native op overlay**, not to the base architecture's
  availability.

## 3. Consumer wheels

| Package | Version |
| --- | --- |
| torchvision | 0.28.0+gfx1013 |
| torchaudio | 2.11.0+gfx1013 |
| torchcodec | 0.17.0+gfx1013 |

ONNX stack: onnx 1.23.2, onnxscript 0.7.2, onnx-ir 1.1.0, onnxruntime 1.31.0,
ml-dtypes 0.6.0, protobuf 7.36.2, flatbuffers 25.12.19. ONNX Runtime GPU requires
an explicit registered execution provider (`bc250-ort`).

## 4. Kernel routes: qualified vs absent

| Route | Status | Notes |
| --- | --- | --- |
| Attention — math/eager SDPA | qualified | reference bring-up path; FP32/BF16/FP16 |
| Attention — tiled/hybrid | performance-qualified | opt-in; causal fwd 1.5–2.9× faster, full attention 12–86× less peak memory (slower compute); dropout/compile/graph boundaries documented; not a default |
| Conv (MIOpen) | qualified | FP16 FP32-accumulation repair; versioned cache |
| Gated-delta | bounded | owned fixtures; not a general FLA guarantee |
| Quantization — W8 | frozen | explicit opt-in, frozen layouts |
| Quantization — NF4 / QLoRA | bounded | native dequantize-to-FP16 opt-in; ~3.56× weight-memory; first derivatives, frozen base, dims ≤4096; no model-quality claim |
| Quantization — packed INT4 | hardware-blocked | native CDNA2/MFMA path absent on RDNA1; TorchAO/bitsandbytes not installed |
| Sparse factorization — ILU(0)/IC(0) + preconditioned solve | bounded | native hipSPARSE csrilu02/csric02 + csrsv2 opt-in; machine precision; RHS gradient; orders ≤256; rocSPARSE has no general direct sparse LU |
| Nonsymmetric eig | host-LAPACK | production default retained; native small-Schur slower for small orders; larger/ill-conditioned separate |
| Compiler (torch.compile) | expanded | dense/gated/CNN/batchnorm + attention/LoRA/embed+CE/dynamic; fullgraph real Inductor, parity vs CPU + eager; correctness coverage, not a throughput claim; eager is the reference |
| Graph capture (HIP) — training step | qualified | opt-in; bitwise-exact LoRA-step capture (fwd+bwd+eager AdamW, whole-step capturable AdamW, checkpointing, fresh-process recapture/resume); ~1.9× launch-bound stack; static shapes, dropout=0 |

A fallback can be correct yet slow. Warning suppression is not a backend fix.

## 5. New-model admission and requalification

Admission criteria (owned, random-init from config, no downloaded weights):

1. import/load on the native provider;
2. GPU forward (+ logits/loss vs an independent CPU/FP64 reference where feasible);
3. GPU backward with finite/nonzero expected gradients;
4. an optimizer step;
5. bitwise checkpoint save/resume and exact resumed continuation;
6. record the **actual** attention/conv/delta/quant routes exercised and any
   fallback, plus dtype/shape/mask/attention-implementation/seed/tolerance.

Admission is bounded validation, not universal correctness or model-quality
proof. Forward support alone is insufficient for a training claim.

**Admitted model families** (owned random-init configs, full battery passed with
exact resume): GPT2 causal-LM, Qwen3, the Qwen3.5 hybrid (gated-delta + conv),
a diffusion-style UNet2D (pure-torch noise-prediction net, not a full diffusion
pipeline), and RNN/LSTM/GRU next-token LMs. Op-level route probes fire on every
run (attention backend flags, tiled-SDPA, gated-delta, causal-conv, frozen-W8).
These establish the architecture/route path, not model quality.

**Requalification rules:**

- A native Torch/ROCm **source change** requires requalification of the affected
  frozen helpers and any admitted model that uses the changed path.
- A **package version change** in any environment requires re-running the
  affected fixtures/admission; the frozen helpers refuse on pin mismatch.
- No model-family promise from an architecture name alone; each model is admitted
  individually with its recorded routes/fallbacks.
- Installing a newer Transformers does not automatically qualify the frozen
  helpers or a new model's kernels.