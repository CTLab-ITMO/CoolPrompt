"""Best-of selection across optimizer methods (portfolio)."""


def select_portfolio(results):
    """Pick the method whose prompt scored highest on val.

    Args:
        results: dict mapping method name -> (prompt, val_score).
            Insertion order breaks ties (first listed wins).

    Returns:
        (method, prompt, val_score) of the winner.
    """
    best = None
    for method, (prompt, score) in results.items():
        if best is None or score > best[2]:
            best = (method, prompt, score)
    return best
