"""Cross-seed robustness analysis: pe2_sgr vs pe2.

Mean accuracy of SGR ~= PE2 (see CONSOLIDATED_FINDINGS.md). This
script asks the orthogonal question: is SGR more *robust* across
random seeds even when its mean is tied? Three robustness axes,
all computed from the final scalar already stored in the logs:

  1. Dispersion  -- cross-seed std of final_metric (lower = steadier)
  2. Worst-case  -- min over seeds (higher = better floor)
  3. Regression  -- fraction of runs where final_metric < start_score
                    (the optimizer made the prompt WORSE)

For each axis we pair SGR vs PE2 per benchmark and report a sign
test over benchmarks. No new runs -- reuses committed logs.

Run: uv run python scripts/variance_analysis.py
"""

import glob
import json
import re
import statistics as st

# model family -> glob patterns for its seed logs
FAMILIES = {
    "gpt-4o-mini": [
        "logs/native_gpt4omini_s*.json",
        "logs/gpt_extra_s*.json",
        "logs/gpt_full_s*.json",
    ],
    "qwen8b": ["logs/ms_qwen8b_s*.json"],
    "qwen14b": ["logs/ms_qwen14b_s*.json"],
    "qwen32b": ["logs/ms_qwen32b_s*.json"],
}

# robustness needs a reasonable sample; skip thin (bench, method) cells
MIN_SEEDS = 5


def _seed(path):
    m = re.search(r"_s(\d+)\.json$", path)
    return m.group(1) if m else "?"


def load(patterns):
    """data[(bench, method)][seed] = (start_score, final_metric)."""
    data = {}
    for pat in patterns:
        for f in glob.glob(pat):
            seed = _seed(f)
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for key, entry in d.items():
                mm = entry.get("metric")
                if not mm or "final_metric" not in mm:
                    continue
                bench, method = key.rsplit("/", 1)
                data.setdefault((bench, method), {})[seed] = (
                    mm.get("start_score"),
                    mm["final_metric"],
                )
    return data


def _finals(cell):
    return [v[1] for v in cell.values()]


def _regressions(cell):
    """count runs where final < start (start may be None -> skip)."""
    bad = tot = 0
    for start, final in cell.values():
        if start is None:
            continue
        tot += 1
        if final < start - 1e-9:
            bad += 1
    return bad, tot


def analyse(name, data, min_seeds):
    benches = sorted({b for (b, _) in data})
    rows = []
    for b in benches:
        sgr = data.get((b, "pe2_sgr"), {})
        pe2 = data.get((b, "pe2"), {})
        shared = sorted(set(sgr) & set(pe2))
        if len(shared) < min_seeds:
            continue
        sf = [sgr[s][1] for s in shared]
        pf = [pe2[s][1] for s in shared]
        rows.append({
            "bench": b,
            "n": len(shared),
            "mean_sgr": st.mean(sf),
            "mean_pe2": st.mean(pf),
            "std_sgr": st.pstdev(sf),
            "std_pe2": st.pstdev(pf),
            "min_sgr": min(sf),
            "min_pe2": min(pf),
        })
    if not rows:
        print(f"\n### {name}: no benchmark with >= {min_seeds} "
              f"paired seeds\n")
        return

    print(f"\n{'='*72}\n### {name}  (paired pe2_sgr vs pe2, "
          f">= {min_seeds} seeds)\n{'='*72}")
    print(f"{'benchmark':24s} {'mean':>13s}  {'std (disp.)':>15s}  "
          f"{'worst-seed':>15s}")
    print(f"{'':24s} {'sgr / pe2':>13s}  {'sgr / pe2':>15s}  "
          f"{'sgr / pe2':>15s}")
    std_win = std_tie = std_loss = 0
    floor_win = floor_tie = floor_loss = 0
    d_std = []
    for r in rows:
        ds = r["std_pe2"] - r["std_sgr"]  # >0 => SGR steadier
        d_std.append(ds)
        if r["std_sgr"] < r["std_pe2"] - 1e-9:
            std_win += 1
        elif r["std_sgr"] > r["std_pe2"] + 1e-9:
            std_loss += 1
        else:
            std_tie += 1
        if r["min_sgr"] > r["min_pe2"] + 1e-9:
            floor_win += 1
        elif r["min_sgr"] < r["min_pe2"] - 1e-9:
            floor_loss += 1
        else:
            floor_tie += 1
        print(
            f"{r['bench']:24s} "
            f"{r['mean_sgr']:.2f}/{r['mean_pe2']:.2f}  "
            f"{r['std_sgr']:6.3f}/{r['std_pe2']:6.3f}  "
            f"{r['min_sgr']:6.2f}/{r['min_pe2']:6.2f}"
        )

    nb = len(rows)
    print(f"\n  benchmarks compared: {nb}")
    print(f"  DISPERSION  SGR steadier (lower std) in "
          f"{std_win}/{nb}  (tie {std_tie}, pe2 {std_loss})")
    print(f"              mean std  sgr={st.mean(r['std_sgr'] for r in rows):.4f}"  # noqa: E501
          f"  pe2={st.mean(r['std_pe2'] for r in rows):.4f}")
    print(f"              mean(std_pe2 - std_sgr) = "
          f"{st.mean(d_std):+.4f}  (>0 => SGR steadier)")
    print(f"  WORST-CASE  SGR higher floor in "
          f"{floor_win}/{nb}  (tie {floor_tie}, pe2 {floor_loss})")
    print(f"              mean worst-seed  sgr="
          f"{st.mean(r['min_sgr'] for r in rows):.3f}"
          f"  pe2={st.mean(r['min_pe2'] for r in rows):.3f}")

    # regression rate (pooled over all runs, not just min_seeds cells)
    sgr_bad = sgr_tot = pe2_bad = pe2_tot = 0
    for b in benches:
        bbad, btot = _regressions(data.get((b, "pe2_sgr"), {}))
        sgr_bad += bbad
        sgr_tot += btot
        bbad, btot = _regressions(data.get((b, "pe2"), {}))
        pe2_bad += bbad
        pe2_tot += btot
    if sgr_tot and pe2_tot:
        print(f"  REGRESSION  runs where final<start: "
              f"sgr={sgr_bad}/{sgr_tot} ({sgr_bad/sgr_tot:.1%})"
              f"  pe2={pe2_bad}/{pe2_tot} ({pe2_bad/pe2_tot:.1%})")


def main():
    print("CROSS-SEED ROBUSTNESS: pe2_sgr vs pe2")
    print("(mean is ~tied per CONSOLIDATED_FINDINGS; this probes "
          "consistency)")
    for name, pats in FAMILIES.items():
        data = load(pats)
        ms = MIN_SEEDS if name == "gpt-4o-mini" else 3
        analyse(name, data, ms)


if __name__ == "__main__":
    main()
