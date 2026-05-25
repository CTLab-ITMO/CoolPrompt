"""Portfolio (best-of PE2 / improved-SGR) + niche map over the
committed 8-seed gpt-4o-mini results.

Uses Phase-1-improved SGR on ifeval (the feedback-on `gate_on`
runs); plain pe2_sgr elsewhere (Phase 1 is verifiable-only).
Selection uses select_portfolio on per-(benchmark, seed) final
scores. Reports per-benchmark portfolio vs pe2 vs sgr and the
fraction of seeds SGR is selected (the niche map).

Run: uv run python scripts/portfolio_niche_map.py
"""

import glob
import json
import statistics as st

from coolprompt.evaluator.portfolio import select_portfolio

GPT_FILES = (
    glob.glob("logs/native_gpt4omini_s*.json")
    + glob.glob("logs/gpt_extra_s*.json")
    + glob.glob("logs/gpt_full_s*.json")
)


def _seed(path):
    return path.split("_s")[-1].replace(".json", "")


def _load(method):
    """(bench, seed) -> final_metric for a method."""
    out = {}
    for f in GPT_FILES:
        s = _seed(f)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for key, e in d.items():
            if not key.endswith("/" + method):
                continue
            m = e.get("metric")
            if m:
                out[(key.rsplit("/", 1)[0], s)] = m["final_metric"]
    return out


def _load_improved_ifeval_sgr():
    """ifeval SGR with Phase-1 feedback (gate_on runs)."""
    out = {}
    for f in glob.glob("logs/gate_on_s*.json"):
        s = _seed(f)
        try:
            m = json.load(open(f)).get(
                "ifeval/pe2_sgr", {}
            ).get("metric")
        except Exception:
            m = None
        if m:
            out[("ifeval", s)] = m["final_metric"]
    return out


def main():
    pe2 = _load("pe2")
    sgr = _load("pe2_sgr")
    sgr.update(_load_improved_ifeval_sgr())  # use improved ifeval

    benches = sorted({b for (b, _) in pe2})
    print("=== Portfolio (best-of pe2 / improved-sgr), 8-seed ===")
    print(f"{'benchmark':26s} {'pe2':>6s} {'sgr':>6s} "
          f"{'portf':>6s}  sgr_win")
    p_all = []
    pe2_all = []
    sgr_all = []
    for b in benches:
        seeds = sorted(
            {s for (bb, s) in pe2 if bb == b}
            & {s for (bb, s) in sgr if bb == b}
        )
        if not seeds:
            continue
        port = []
        sgr_wins = 0
        for s in seeds:
            res = {
                "pe2": ("", pe2[(b, s)]),
                "pe2_sgr": ("", sgr[(b, s)]),
            }
            winner, _, score = select_portfolio(res)
            port.append(score)
            if winner == "pe2_sgr":
                sgr_wins += 1
        mp = st.mean(pe2[(b, s)] for s in seeds)
        ms = st.mean(sgr[(b, s)] for s in seeds)
        mo = st.mean(port)
        p_all.append(mo)
        pe2_all.append(mp)
        sgr_all.append(ms)
        print(f"{b:26s} {mp:6.2f} {ms:6.2f} {mo:6.2f}  "
              f"{sgr_wins}/{len(seeds)}")
    print()
    print(f"OVERALL mean  pe2={st.mean(pe2_all):.3f}  "
          f"sgr={st.mean(sgr_all):.3f}  "
          f"portfolio={st.mean(p_all):.3f}")


if __name__ == "__main__":
    main()
