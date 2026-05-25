"""Aggregate gpt-4o-mini multi-seed results into mean+-std and a
paired pe2_sgr-vs-pe2 sign test per benchmark.

Pulls final_metric for every (benchmark, method, seed) from all
gpt-4o-mini result files:
  - logs/native_gpt4omini_s<seed>.json  (orig 8 benchmarks, seeds 42/123/2024)
  - logs/gpt_extra_s<seed>.json         (new 9 benchmarks, seeds 42/123/2024)
  - logs/gpt_full_s<seed>.json          (all 17 benchmarks, extra seeds)

Run: uv run python scripts/aggregate_gpt_seeds.py
"""

import glob
import json
import re
import statistics as st

METHODS = ["pe2", "pe2_sgr", "ape", "opro"]


def _seed(path):
    m = re.search(r"_s(\d+)\.json$", path)
    return m.group(1) if m else "?"


def load():
    # data[(bench, method)][seed] = final_metric
    data = {}
    benches = set()
    patterns = [
        "logs/native_gpt4omini_s*.json",
        "logs/gpt_extra_s*.json",
        "logs/gpt_full_s*.json",
    ]
    for pat in patterns:
        for f in glob.glob(pat):
            seed = _seed(f)
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for key, entry in d.items():
                mm = entry.get("metric")
                if not mm:
                    continue
                bench, method = key.rsplit("/", 1)
                benches.add(bench)
                data.setdefault((bench, method), {})[seed] = (
                    mm["final_metric"]
                )
    return data, sorted(benches)


def sign_test_better(a, b):
    """Fraction of paired seeds where a > b / == / < b."""
    wins = sum(1 for x, y in zip(a, b) if x > y)
    ties = sum(1 for x, y in zip(a, b) if x == y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    return wins, ties, losses


def main():
    data, benches = load()
    print("=== gpt-4o-mini multi-seed aggregate (final_metric) ===")
    hdr = f"{'benchmark':26s} " + " ".join(
        f"{m:>13s}" for m in METHODS
    )
    print(hdr)
    means = {m: [] for m in METHODS}
    for b in benches:
        cells = []
        for m in METHODS:
            vals = list(data.get((b, m), {}).values())
            if not vals:
                cells.append("--")
            else:
                mu = st.mean(vals)
                sd = st.pstdev(vals) if len(vals) > 1 else 0.0
                cells.append(f"{mu:.2f}+-{sd:.2f}(n{len(vals)})")
                means[m].append(mu)
        print(f"{b:26s} " + " ".join(f"{c:>13s}" for c in cells))

    print("\n=== overall mean final (across benchmarks) ===")
    for m in METHODS:
        if means[m]:
            print(f"  {m:8s} {st.mean(means[m]):.3f}")

    print("\n=== paired pe2_sgr vs pe2 per benchmark "
          "(shared seeds) ===")
    for b in benches:
        sgr = data.get((b, "pe2_sgr"), {})
        pe2 = data.get((b, "pe2"), {})
        shared = sorted(set(sgr) & set(pe2))
        if not shared:
            continue
        a = [sgr[s] for s in shared]
        c = [pe2[s] for s in shared]
        w, t, ls = sign_test_better(a, c)
        diff = st.mean(a) - st.mean(c)
        verdict = (
            "SGR>" if diff > 0 else "SGR<" if diff < 0 else "tie"
        )
        print(
            f"  {b:26s} n={len(shared)} "
            f"sgr-pe2={diff:+.3f} W/T/L={w}/{t}/{ls} {verdict}"
        )


if __name__ == "__main__":
    main()
