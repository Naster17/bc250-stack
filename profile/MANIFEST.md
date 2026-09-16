# Deployed profile snapshot (2026-09-16)

Source of truth for deployment remains `~/bc250work/bc250-rocm-profile`
(branch `main`) on the board. This file pins what r1.2 validated.

- Profile commits: `fc26ae8` (r1.2 promotion), `b008c05` (wheel hashes).
- `config.env`: `RUNTIME_IMAGE=localhost/bc250-rocm-runtime:7.2.4-gfx1013-r1.2`,
  `TORCH_VENV=/opt/torch-venv`.
- r1.2 image id `cd9f0f9bd955236a8fd7d4095c6b256d2d688caf7d8b8c92314754037a51f634`.
- Wheel `torch-2.9.1a0+gitunknown-cp312-cp312-linux_x86_64.whl`
  `abe93cd86b33a595b52f290e050a925fc4ac04e4746fb8f7f9bcefdfb8aa08bc`.
- `libhipblas.so.3.5` (solver rebuild) `bff57f54929e298753fc52b7f5fa165bd0bf6dbdb5fe012c29ad0db592553c04`.
- Research handoff `BC250-RESEARCH.md` `fb26c8725176b7bac46caae64a32b8aae954a074fabb98c9b3e834ae4eb48873`.
