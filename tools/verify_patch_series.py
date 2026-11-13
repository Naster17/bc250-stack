#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Patch-apply reconstruction proof for the published BC-250 component forks.

For each component in ci/components.json: fetch the recorded upstream base, apply
the umbrella patch series, and verify the resulting tree exactly matches the
published fork's tree (expected_tree). This proves the published patch series
applies cleanly to the upstream base and reproduces the published fork content.
Git-only; no ROCm toolchain or build is required, so it runs on ordinary GitHub
Actions runners.

Two apply modes (manifest key "apply", default "am"):

* "am"    — checkout the base and `git am` the series. Used for small bases.
* "index" — no checkout: fetch the base with --filter=blob:none (trees/commits
  only), `git read-tree` it, `git apply --cached` each patch (lazy-fetching only
  the patched files' preimage blobs), then `git write-tree --missing-ok` with
  lazy fetch disabled. Used for LFS-heavy bases whose full blob set is far too
  large to check out on a runner (rocm-libraries' base tree is ~9.4 GiB of git
  blobs; the index mode transfers only a few MiB). The resulting tree hash is
  identical to what `git am` would produce.

Usage:
    python3 tools/verify_patch_series.py                 # all components
    python3 tools/verify_patch_series.py --component triton
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

# Never download LFS blobs during verification: the reconstruction proof is
# about git tree identity, and LFS pointer files are the tracked content.
BASE_ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")


def run(cmd, cwd=None, env=None):
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, env=env or BASE_ENV
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed: "
            + " ".join(str(c) for c in cmd)
            + "\n"
            + proc.stdout[-1500:]
            + "\n"
            + proc.stderr[-1500:]
        )
    return proc.stdout.strip()


def verify(name, cfg, umb):
    patches = sorted((umb / cfg["patches"]).glob("*.patch"))
    if len(patches) != cfg["patch_count"]:
        raise RuntimeError(
            f"patch count {len(patches)} != manifest {cfg['patch_count']}"
        )
    mode = cfg.get("apply", "am")
    if mode not in ("am", "index"):
        raise RuntimeError(f"unknown apply mode {mode!r}")
    with tempfile.TemporaryDirectory(prefix=f"bc250-ci-{name}-") as tmp:
        run(["git", "init", "-q", tmp])
        run(["git", "-C", tmp, "config", "user.email", "ci@bc250.local"])
        run(["git", "-C", tmp, "config", "user.name", "bc250-ci"])
        # Fetch only the recorded base commit (GitHub allows reachability-free
        # SHA fetch); no full upstream history is downloaded.
        fetch = ["git", "-C", tmp, "fetch", "--depth", "1"]
        if mode == "index":
            fetch.append("--filter=blob:none")
        run(fetch + [cfg["upstream"], cfg["base"]])
        if mode == "am":
            run(["git", "-C", tmp, "checkout", "-q", "FETCH_HEAD"])
            run(["git", "-C", tmp, "am", *[str(p) for p in patches]])
            tree = run(["git", "-C", tmp, "rev-parse", "HEAD^{tree}"])
        else:
            run(["git", "-C", tmp, "read-tree", "FETCH_HEAD^{tree}"])
            for patch in patches:
                run(["git", "-C", tmp, "apply", "--cached", str(patch)])
            # --missing-ok: base blobs were never fetched (blob:none); the tree
            # hash only needs the index entries. Disable lazy fetch so
            # write-tree cannot pull the full base blob set.
            tree = run(
                ["git", "-C", tmp, "write-tree", "--missing-ok"],
                env=dict(BASE_ENV, GIT_NO_LAZY_FETCH="1"),
            )
        return tree == cfg["expected_tree"], tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", help="verify a single component (default: all)")
    parser.add_argument("--manifest", default=str(ROOT / "ci" / "components.json"))
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    components = manifest["components"]
    if args.component and args.component not in components:
        parser.error(f"unknown component {args.component}")
    names = [args.component] if args.component else list(components)
    failed = []
    for name in names:
        cfg = components[name]
        try:
            ok, tree = verify(name, cfg, ROOT)
            label = "PASS" if ok else "FAIL(tree-mismatch)"
            print(
                f"[{label}] {name}: {cfg['patch_count']} patches -> tree {tree[:12]} "
                f"(expected {cfg['expected_tree'][:12]})",
                flush=True,
            )
            if not ok:
                failed.append(name)
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"[FAIL] {name}: {error}", flush=True)
            failed.append(name)
    if failed:
        print(f"FAILED: {', '.join(failed)}", flush=True)
        sys.exit(1)
    print(f"ALL_PASS ({len(names)} components)", flush=True)


if __name__ == "__main__":
    main()
