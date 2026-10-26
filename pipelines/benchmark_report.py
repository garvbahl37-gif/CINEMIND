"""
Write docs/BENCHMARK.md from the experiment records in docs/benchmarks/.

    python pipelines/benchmark_report.py [--dir docs/benchmarks] [--out docs/BENCHMARK.md]

Every number in the report is read from a JSON file that pipelines/evaluate.py
wrote; none is typed in by hand, so the report cannot drift from the runs.
Before/after differences are paired: both runs scored the same people on the
same held-out ratings, and each person's score is compared with their own.

The runs it expects (see the commands in the report):
  before        the catalogue as deployed before the expansion
  after-common  the expanded system, candidates and targets limited to the
                films of the old catalogue (same people, same test set)
  after-full    the expanded system over its whole catalogue
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

NAMES = {
    "popularity": "Most popular",
    "item-cf": "Item-item CF",
    "two-tower": "Two-tower alone",
    "adaptive": "**Hybrid (adaptive engine)**",
    "adaptive − drift": "Hybrid − drift",
    "adaptive − diversity": "Hybrid − diversity",
    "adaptive − exploration": "Hybrid − exploration",
    "adaptive − negatives": "Hybrid − negatives",
}
MAIN = ["popularity", "item-cf", "two-tower", "adaptive"]


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def ci(c):
    return f"{c[0]:.3f}–{c[1]:.3f}"


def signed_ci(c):
    return f"{100 * c[0]:+.2f} to {100 * c[1]:+.2f}"


def load(d: Path, label: str):
    f = d / f"{label}.json"
    if not f.exists():
        return None, None
    run = json.loads(f.read_text())
    pu = d / f"{label}.per_user.json.gz"
    per_user = json.loads(gzip.decompress(pu.read_bytes())) if pu.exists() else None
    return run, per_user


def paired(a: dict, b: dict, metric: str, policy: str, seed: int = 7, n: int = 2000):
    """Mean of b − a over the people both runs scored, with a 95% bootstrap interval."""
    ia = {u: k for k, u in enumerate(a["users"])}
    common = [u for u in b["users"] if u in ia]
    ib = {u: k for k, u in enumerate(b["users"])}
    x = np.array([a[metric][policy][ia[u]] for u in common])
    y = np.array([b[metric][policy][ib[u]] for u in common])
    d = y - x
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], len(common)


def experiments_table(runs: dict) -> list[str]:
    rows = ["| Run | Films in catalogue | Candidate films | Films with training ratings | People | Test positives "
            "| Seed | Code | Timestamp |",
            "|---|---|---|---|---|---|---|---|---|"]
    for label, run in runs.items():
        e = run["experiment"]
        code = e["code"]["commit"] + (" + uncommitted changes" if e["code"]["uncommitted_changes"] else "")
        rows.append(f"| `{label}` | {e['catalog_films']:,} | {e['candidate_films']:,} | "
                    f"{e['films_with_training_ratings']:,} | {e['evaluated_users']:,} | {e['test_interactions']:,} | "
                    f"{e['seed']} | `{code}` | {e['timestamp']} |")
    return rows


def accuracy_table(m: dict) -> list[str]:
    rows = ["| Policy | Recall@10 | Recall@20 (95% CI) | Precision@10 | Precision@20 | NDCG@10 | NDCG@20 | MAP@20 "
            "| Hit rate@10 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in NAMES:
        if k not in m:
            continue
        x = m[k]
        rows.append(f"| {NAMES[k]} | {f3(x['recall_10'])} | {f3(x['recall'])} ({ci(x['recall_ci'])}) | "
                    f"{f3(x['precision_10'])} | {f3(x['precision_20'])} | {f3(x['ndcg_10'])} | {f3(x['ndcg'])} | "
                    f"{f3(x['map_20'])} | {pct(x['hit_rate_10'])} |")
    return rows


def beyond_table(m: dict) -> list[str]:
    rows = ["| Policy | Coverage | Diversity (ILD) | Novelty (bits) | Beyond top 1,000 | Cold-start share "
            "| Cold recall@20 (people) | Warm recall@20 | Latency p50 / p95 / p99 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in NAMES:
        if k not in m:
            continue
        x = m[k]
        lat = x["latency_ms"]
        cold = f"{f3(x['cold_recall'])} ({x['cold_recall_users']:,})" if x["cold_recall_users"] else "— (0)"
        rows.append(f"| {NAMES[k]} | {pct(x['coverage'])} | {f3(x['diversity'])} | {x['novelty']:.2f} | "
                    f"{pct(x['long_tail'])} | {pct(x['cold_start_share'])} | {cold} | {f3(x['warm_recall'])} | "
                    f"{lat['p50']:.2f} / {lat['p95']:.2f} / {lat['p99']:.2f} ms |")
    return rows


def ablation_table(m: dict) -> list[str]:
    rows = ["| Switched off | Recall@20 change (paired 95% CI, points) | NDCG@20 change (points) | Reading |",
            "|---|---|---|---|"]
    for k in NAMES:
        if not k.startswith("adaptive − ") or k not in m:
            continue
        v = m[k]["vs_adaptive"]
        lo, hi = v["recall_ci"]
        reading = ("no measurable effect" if lo <= 0 <= hi else
                   "the part helps" if hi < 0 else "the part costs recall offline")
        rows.append(f"| {k.split('− ')[1]} | {100 * v['recall']:+.2f} ({signed_ci(v['recall_ci'])}) | "
                    f"{100 * v['ndcg']:+.2f} | {reading} |")
    return rows


def tower_line(run: dict) -> str:
    t = run["two_tower"]
    if not t.get("trained"):
        return "original item vectors (no trained user tower)"
    protocol = ("all of each person's 2015 likes, from their likes before 2015" if t.get("selection") else
                "each person's last like before 2016, held out")
    parts = [f"validation Recall@20 {t['validation_recall@20']:.3f} from a full history, "
             f"{t['validation_recall@20_last3']:.3f} from the last three likes — predicting {protocol}"]
    if "films" in t:
        parts.append(f"{t['films']:,} films, {t.get('films_with_training_data', 0):,} with training data")
    if t.get("uniform_negatives"):
        parts.append(f"{t['uniform_negatives']:,} uniform negatives per batch")
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "docs" / "benchmarks")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "BENCHMARK.md")
    args = ap.parse_args()

    runs, per_user = {}, {}
    for label in ("before", "after-common", "after-full"):
        run, pu = load(args.dir, label)
        if run:
            runs[label], per_user[label] = run, pu
    if not runs:
        raise SystemExit(f"no experiment records in {args.dir}")
    rel = args.dir.relative_to(ROOT) if args.dir.is_relative_to(ROOT) else args.dir

    L = ["# Benchmark", "",
         f"Generated by `pipelines/benchmark_report.py` from the experiment records in `{rel}/`. "
         "Do not edit by hand — rerun the evaluations and the generator.", ""]

    L += ["## Reproduce", "", "```bash",
          "# 1. choose the adaptive engine's blend on 2015 (data/raw/validation comes from rebuild.py)",
          "python pipelines/evaluate.py --ml32m data/raw/ml-32m --art data/raw/validation \\",
          "  --cutoff 2015-01-01 --until 2016-01-01 --users 3000 --seed 42 --no-ope --label validation \\",
          f"  --blend-grid {rel}/hybrid_grid.json --out {rel}/validation.json",
          "# 2. the test period, before and after, on the same people",
          "python pipelines/evaluate.py --ml32m data/raw/ml-32m --art data/raw/baseline-artifacts \\",
          f"  --users 3000 --seed 42 --no-ope --label before --save-users {rel}/users.json \\",
          f"  --out {rel}/before.json --per-user {rel}/before.per_user.json.gz",
          "python pipelines/evaluate.py --ml32m data/raw/ml-32m --art data/artifacts \\",
          f"  --universe data/raw/baseline-artifacts/catalog.json --users-file {rel}/users.json \\",
          f"  --no-ope --label after-common --out {rel}/after-common.json \\",
          f"  --per-user {rel}/after-common.per_user.json.gz",
          "python pipelines/evaluate.py --ml32m data/raw/ml-32m --art data/artifacts \\",
          f"  --users-file {rel}/users.json --label after-full --out {rel}/after-full.json \\",
          f"  --per-user {rel}/after-full.per_user.json.gz",
          "cp " + f"{rel}/after-full.json data/artifacts/eval.json        # what /api/evaluation serves",
          "python pipelines/benchmark_report.py",
          "```", "",
          "`data/raw/baseline-artifacts` is `data/artifacts` as it was deployed before the expansion "
          "(git commit in its `COMMIT` file).", ""]

    L += ["## Runs", ""] + experiments_table(runs) + [""]
    ds = next(iter(runs.values()))["experiment"]["dataset"]
    if ds.get("checksums"):
        sums = ds["checksums"]
        pairs = ", ".join(f"`{f}` {h}" for h, f in zip(sums[::2], sums[1::2]))
        L += [f"Dataset: {ds['name']}, MD5 as published by GroupLens: {pairs}.", ""]
    for label, run in runs.items():
        L.append(f"- `{label}` two-tower model: {tower_line(run)}.")
    L += ["", "The two validation protocols differ — the newer one predicts a whole year of likes, the older one a "
          "single held-out like — so their numbers are not comparable with each other; the test-period results "
          "below are.", ""]

    sel = next((r["two_tower"].get("selection") for r in reversed(list(runs.values()))
                if r["two_tower"].get("selection")), None)
    if sel and sel.get("grid"):
        L += ["## Two-tower model selection", "",
              f"Chosen on validation only — {sel['validation']}. Films first liked in 2015 are scored by content "
              "alone, as a new film is served. ID dropout trains the model to score a film from its content on the "
              "same scale as one it knows; without it, films with no training data crowd the lists.", "",
              "| ID dropout | Best epoch | Recall@20, full history | last three likes | New-film recall@20 "
              "| New films' share of top-20 slots |",
              "|---|---|---|---|---|---|"]
        for g in sel["grid"]:
            mark = "**" if g["id_dropout"] == sel["id_dropout"] else ""
            L.append(f"| {mark}{g['id_dropout']}{mark} | {g['epoch']} | {f3(g['recall@20'])} | "
                     f"{f3(g['recall@20_last3'])} | {f3(g['new_film_recall@20'])} | "
                     f"{pct(g['new_film_share_of_top20'])} |")
        L += ["", f"The model is then refit on every like before 2016 with ID dropout {sel['id_dropout']} "
              f"for {sel['epoch']} epochs.", ""]

    val, _ = load(args.dir, "validation")
    if val:
        sys.path.insert(0, str(ROOT / "api"))
        from _adaptive import Options
        served = {k: v for k, v in vars(Options()).items() if not isinstance(v, bool)}
        current = val["experiment"].get("adaptive_options") or served
        m = val["metrics"]
        blends = [(k, json.loads(k[len("blend "):])) for k in m if k.startswith("blend ")]
        e = val["experiment"]
        cut, until, tcut = e["split"]["cutoff"], e["split"].get("until"), val["two_tower"].get("cutoff")
        if not until or (tcut or "9999") > cut:
            raise SystemExit(f"validation.json is not a validation run: test period {cut}–{until}, "
                             f"two-tower trained up to {tcut}")
        L += ["## Hybrid selection", "",
              f"The adaptive engine's blend weights and similarity thresholds are chosen on validation, never on "
              f"the test period: the whole "
              f"harness run with {cut} – {until} as the held-out period (`--cutoff {cut} --until {until}`), the "
              f"two-tower model trained on likes before {tcut}, {e['evaluated_users']:,} people and "
              f"{e['test_interactions']:,} held-out likes. Rule, fixed before the run: a setting replaces the current "
              "one only if its paired 95% interval against it excludes zero.", "",
              "| Setting | Recall@20 | NDCG@20 | Change vs current (points, paired 95% CI) |",
              "|---|---|---|---|",
              f"| current {json.dumps(current)} | {f3(m['adaptive']['recall'])} | {f3(m['adaptive']['ndcg'])} | — |"]
        winners = []
        for k, b in blends:
            v = m[k]["vs_adaptive"]
            L.append(f"| {json.dumps(b)} | {f3(m[k]['recall'])} | {f3(m[k]['ndcg'])} | "
                     f"{100 * v['recall']:+.2f} ({signed_ci(v['recall_ci'])}) |")
            if v["recall_ci"][0] > 0:
                winners.append((m[k]["recall"], b))
        if winners:
            pick = max(winners, key=lambda w: w[0])[1]
            verdict = f"Chosen: {json.dumps(pick)}."
            ok = all(abs(served[k] - v) < 1e-9 for k, v in {**current, **pick}.items())
        else:
            verdict = "No setting beats the current ones; they stay."
            ok = all(abs(served[k] - v) < 1e-9 for k, v in current.items())
        chance = len(blends) * 0.05
        L += ["", verdict + (" The served defaults (`api/_adaptive.py`, `Options`) match." if ok else
                             " **The served defaults do not match the chosen setting.**"),
              "", f"With {len(blends)} settings tried, about {chance:.1f} intervals would exclude zero by chance "
              "alone, so a winner's size matters as much as its interval: one that changes Recall@20 by a few "
              "hundredths of a point is adopted because the rule, fixed in advance, says so — not because it "
              "matters.", ""]

    L += ["## Method", "",
          "- **Split.** Temporal, at 2016-01-01, on MovieLens 32M. Ratings before it are the only training data "
          "for every model — item-item CF, popularity and the quality prior are rebuilt from them here, and the "
          "two-tower model was trained on them, with its epoch chosen on a validation set inside the training "
          "period. Ratings after it are the test set and are never used for selection.",
          "- **People.** Drawn once (seed 42) from those with ≥ 10 positives (≥ 4★) before the cutoff and ≥ 3 "
          "positives after it on films they hadn't rated; the same people are used in every run "
          f"(`{rel}/users.json`). Their training ratings are the input, with their real timestamps.",
          "- **Targets.** A later ≥ 4★ rating on a film the person hadn't rated is a hit. Films already rated "
          "are never recommended.",
          "- **Common universe.** `after-common` restricts both candidates and targets to the films of the old "
          "catalogue, so it has exactly the test set of `before` and the two can be compared person by person. "
          "`after-full` scores the whole expanded catalogue; its test set is larger, so its accuracy is *not* "
          "comparable with `before`.",
          "- **Metrics.** Recall@K = hits in the top K / min(|targets|, K). Precision@K = hits / K. NDCG@K with "
          "binary gains. MAP@20 = mean average precision, normalised by min(|targets|, 20). Coverage = distinct "
          "films recommended / candidate films. Diversity = mean pairwise (1 − cosine) of the films' content "
          "vectors within a list. Novelty = mean −log2 of a film's share of training users who liked it. "
          "Cold-start share = recommended films with no training ratings. Cold / warm recall = Recall@20 on the "
          "targets without / with training ratings. Latency = time to rank one person's list in this harness "
          "(single thread, warm), not HTTP latency.",
          "- **Intervals.** 95% percentile bootstrap over people (1,000 resamples within a run, 2,000 for the "
          "paired before/after differences).", ""]

    for label, run in runs.items():
        title = {"before": "Before the expansion",
                 "after-common": "After, on the old catalogue's films (common universe)",
                 "after-full": "After, over the whole expanded catalogue"}[label]
        m = run["metrics"]
        L += [f"## {title} — `{label}`", "", run["dataset"] + ".", ""]
        L += accuracy_table(m) + [""] + beyond_table(m) + [""]
        if any(k.startswith("adaptive − ") for k in m):
            L += ["Ablations, each part of the hybrid switched off for the same people:", ""]
            L += ablation_table(m) + [""]
        if run.get("ope"):
            o = run["ope"]
            L += ["Counterfactual evaluation — could logs from one recommender say how another would do? "
                  f"{o['logs']:,} simulated impressions from a known logging policy ({o['logging_policy'].lower()}). "
                  f"{o['target_policies']}. The truth is computed exactly, so each estimator can be checked:", "",
                  "| Policy | Truth | IPS | SNIPS | Direct method | Doubly robust (95% CI) | Effective logs |",
                  "|---|---|---|---|---|---|---|"]
            for k in ("popularity", "item-cf", "two-tower", "adaptive"):
                x = o["estimates"][k]
                L.append(f"| {NAMES[k]} | {f3(x['truth'])} | {f3(x['ips'])} | {f3(x['snips'])} | {f3(x['dm'])} | "
                         f"{f3(x['dr'])} ({ci(x['dr_ci'])}) | {x['ess']:,.0f} |")
            tau = o["kendall_tau_vs_truth"]
            L += ["", "Kendall's τ between each estimator's ranking of the four policies and the true one: "
                  + ", ".join(f"{k.upper() if k != 'snips' else 'SNIPS'} {v:+.2f}" for k, v in tau.items()) + ".", ""]

    if "before" in runs and "after-common" in runs and per_user.get("before") and per_user.get("after-common"):
        a, b = per_user["before"], per_user["after-common"]
        cand = {k: runs[k]["experiment"]["candidate_films"] for k in ("before", "after-common")}
        note = ("" if cand["before"] == cand["after-common"] else
                f" (the expanded catalogue offers {cand['after-common']:,} of the {cand['before']:,}: the others are "
                "duplicate rows it keeps only as aliases)")
        L += ["## Before vs after, person by person", "",
              f"Same people, same held-out ratings, same candidate films{note}. A positive change means the "
              "expanded system ranks those people's later favourites higher.", "",
              "| Policy | Recall@20 before | after | Change (points, paired 95% CI) | NDCG@20 change (points, CI) "
              "| Reading |",
              "|---|---|---|---|---|---|"]
        for k in MAIN:
            if k not in a["recall_20"] or k not in b["recall_20"]:
                continue
            dr, cr, n_ = paired(a, b, "recall_20", k)
            dn, cn, _ = paired(a, b, "ndcg_20", k)
            reading = "no measurable change" if cr[0] <= 0 <= cr[1] else ("better" if cr[0] > 0 else "worse")
            L.append(f"| {NAMES[k]} | {f3(runs['before']['metrics'][k]['recall'])} | "
                     f"{f3(runs['after-common']['metrics'][k]['recall'])} | {100 * dr:+.2f} ({signed_ci(cr)}) | "
                     f"{100 * dn:+.2f} ({signed_ci(cn)}) | {reading} |")
        L += ["", f"{n_:,} people in both runs.", ""]

    L += ["## What these numbers cannot tell you", ""]
    seen = set()
    for run in runs.values():
        for s in run.get("assumptions", []) + run.get("notes", []):
            if s not in seen:
                seen.add(s)
                L.append(f"- {s}")
    L += ["- The before/after difference is the change of the whole system: a retrained two-tower model, new "
          "content vectors for added films, and — because the catalogue now holds more of the films people "
          "rated — longer training histories for some of them. It does not isolate any one of these.",
          "- Popularity, item-item CF and the quality prior see only films with training ratings, so films added "
          "without ratings can only be reached through content: the two-tower model's content projection and "
          "the sentence embeddings.", ""]

    args.out.write_text("\n".join(L))
    print(f"[report] wrote {args.out} from {', '.join(runs)}")


if __name__ == "__main__":
    main()
