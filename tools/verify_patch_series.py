#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Patch-apply reconstruction proof for the published BC-250 component forks.

For each component in ci/components.json: fetch the recorded upstream base, apply
the umbrella patch series with `git am`, and verify the resulting tree exactly
matches the published fork's tree (expected_tree). This proves the published patch
series applies cleanly to the upstream base and reproduces the published fork
content. Git-only; no ROCm toolchain or build is required, so it runs on ordinary
GitHub Actions runners.

Usage:
    python3 tools/verify_patch_series.py                 # all components
    python3 tools/verify_patch_series.py --component triton
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed: " + " ".join(str(c) for c in cmd) + "\n"
            + proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]
        )
    return proc.stdout.strip()


def verify(name, cfg, umb):
    patches = sorted((umb / cfg["patches"]).glob("*.patch"))
    if len(patches) != cfg["patch_count"]:
        raise RuntimeError(
            f"patch count {len(patches)} != manifest {cfg['patch_count']}"
        )
    with tempfile.TemporaryDirectory(prefix=f"bc250-ci-{name}-") as tmp:
        run(["git", "init", "-q", tmp])
        run(["git", "-C", tmp, "config", "user.email", "ci@bc250.local"])
        run(["git", "-C", tmp, "config", "user.name", "bc250-ci"])
        # Fetch only the recorded base commit (GitHub allows reachability-free
        # SHA fetch); no full upstream history is downloaded.
        run(["git", "-C", tmp, "fetch", "--depth", "1", cfg["upstream"], cfg["base"]])
        run(["git", "-C", tmp, "checkout", "-q", "FETCH_HEAD"])
        run(["git", "-C", tmp, "am", *[str(p) for p in patches]])
        tree = run(["git", "-C", tmp, "rev-parse", "HEAD^{tree}"])
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
