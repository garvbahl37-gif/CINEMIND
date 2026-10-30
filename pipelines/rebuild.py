"""
Rebuild every serving artifact into a staging directory, validate, then promote.

    python pipelines/rebuild.py --ml32m data/raw/ml-32m --model-dir data/raw/potion-base-8M [--promote]

Steps, each into the staging copy (default data/raw/staging):
  1. build_catalog.py     catalogue, id map, provenance, data-quality report
     build_search_index.py the API's search index, prebuilt from that catalogue
  2. build_semantic.py    sentence embeddings (cached by text, so only new or
                          changed films are encoded)
  3. train_two_tower.py   two-tower model over the whole catalogue, selected on
                          2015; the selected model as trained before 2015 too
  4. build_neighbors.py   24 matches per film, with their basis, and browse rows
  5. validate_artifacts.py

It also assembles data/raw/validation: the staged artifacts with the
before-2015 two-tower model, which pipelines/evaluate.py --cutoff 2015-01-01
uses to choose the adaptive engine's settings without touching the test period.

data/artifacts is untouched until --promote, which validates the staging copy
once more, replaces each changed file with an atomic rename, and removes any
file the build no longer makes. A failed step stops the run and leaves the live
artifacts as they were.

Fetch the inputs first (both resumable): ingest_wikidata.py and, with a TMDB
key, ingest_tmdb.py.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"
PIPE = ROOT / "pipelines"


def run(*cmd: str):
    print(f"\n[rebuild] $ {' '.join(cmd)}", flush=True)
    t = time.time()
    if subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode:
        sys.exit(f"[rebuild] step failed: {cmd[0]} — data/artifacts not touched")
    print(f"[rebuild] ok in {time.time() - t:.0f}s", flush=True)


def promote(stage: Path):
    """Make data/artifacts the staged build: changed files replaced one atomic rename at a
    time, then files the build no longer makes (catalog.json, since it is compressed) removed."""
    changed, removed = [], []
    for f in sorted(stage.iterdir()):
        if not f.is_file() or f.suffix == ".tmp":
            continue
        dest = ART / f.name
        if dest.exists() and filecmp.cmp(f, dest, shallow=False):
            continue
        tmp = ART / (f.name + ".promote.tmp")
        shutil.copy2(f, tmp)
        os.replace(tmp, dest)
        changed.append(f.name)
    for f in sorted(ART.iterdir()):
        if f.is_file() and not (stage / f.name).exists():
            f.unlink()
            removed.append(f.name)
    print(f"[rebuild] promoted {len(changed)} files: {', '.join(changed) or 'none'}"
          + (f"; removed {', '.join(removed)}" if removed else ""))


def assemble_validation(stage: Path, tower_dir: Path, out: Path):
    """The staged artifacts, with the two-tower model trained before 2015 in place of the served one."""
    out.mkdir(parents=True, exist_ok=True)
    for f in stage.iterdir():
        if f.is_file() and f.suffix != ".tmp":
            shutil.copy2(f, out / f.name)
    manifest = json.loads((out / "manifest.json").read_text())
    side = json.loads((tower_dir / "manifest.json").read_text())["artifacts"]
    for name in ("tower_items.npy", "tower_user.npz"):
        shutil.copy2(tower_dir / name, out / name)
        manifest["artifacts"][name] = side[name]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[rebuild] validation artifacts (two-tower trained before 2015) in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ml32m", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True, help="Model2Vec potion-base-8M")
    ap.add_argument("--stage", type=Path, default=ROOT / "data" / "raw" / "staging")
    ap.add_argument("--promote", action="store_true", help="replace data/artifacts with the validated build")
    ap.add_argument("--skip", default="", help="comma-separated steps to skip: catalog,semantic,tower,neighbors")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    args.stage.mkdir(parents=True, exist_ok=True)
    if "catalog" not in skip:
        # Start from the live artifacts: the catalogue is extended, never rebuilt from nothing.
        for f in ART.iterdir():
            if f.is_file():
                shutil.copy2(f, args.stage / f.name)
        run(str(PIPE / "build_catalog.py"), "--ml32m", str(args.ml32m), "--out", str(args.stage),
            "--base", str(args.stage))
        run(str(PIPE / "build_search_index.py"), "--art", str(args.stage))
    if "semantic" not in skip:
        run(str(PIPE / "build_semantic.py"), "--model-dir", str(args.model_dir), "--art", str(args.stage))
    tower_2015 = args.stage.parent / "validation-tower"
    if "tower" not in skip:
        run(str(PIPE / "train_two_tower.py"), "--ml32m", str(args.ml32m), "--art", str(args.stage),
            "--export-selection", str(tower_2015))
    if "neighbors" not in skip:
        run(str(PIPE / "build_neighbors.py"), "--ml32m", str(args.ml32m), "--art", str(args.stage))
    if "tower" not in skip:                 # only a model trained in this run matches this catalogue
        assemble_validation(args.stage, tower_2015, args.stage.parent / "validation")
    run(str(PIPE / "validate_artifacts.py"), "--art", str(args.stage))
    if args.promote:
        run(str(PIPE / "validate_artifacts.py"), "--art", str(args.stage))
        promote(args.stage)
    else:
        print(f"\n[rebuild] staged and validated in {args.stage}; rerun with --promote to serve it")


if __name__ == "__main__":
    main()
