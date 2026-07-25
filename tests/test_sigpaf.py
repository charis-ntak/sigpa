"""Tests for the SIGPAF fuzzy variant (JMSE 2021, 9(11), 1243)."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigpa import Graph, MamdaniFIS, TSKFIS, sigpaf, usv_energy
from sigpa.fuzzy import INPUT_SETS, OUTPUT_SETS, RULES, rank_routes


def test_rule_base_complete():
    """Table 1 has all 27 antecedent combinations, each exactly once."""
    combos = {antecedents for antecedents, _ in RULES}
    assert len(RULES) == 27
    assert len(combos) == 27


def test_membership_functions_match_figures():
    """Spot-check the breakpoints of Figures 3-6."""
    low, med, high = INPUT_SETS["low"], INPUT_SETS["medium"], INPUT_SETS["high"]
    assert low(0.0) == 1.0 and low(0.25) == 1.0 and low(0.5) == 0.0
    assert med(0.25) == 0.0 and med(0.5) == 1.0 and med(0.75) == 0.0
    assert high(0.5) == 0.0 and high(0.75) == 1.0 and high(1.0) == 1.0
    vh = OUTPUT_SETS["very_high"]
    assert vh(5 / 6) == 1.0 and vh(1.0) == 1.0 and vh(2 / 3) == 0.0


def test_fis_extremes_and_monotonicity():
    """Ideal inputs give high quality, worst inputs low, both variants."""
    for fis in (MamdaniFIS(), TSKFIS()):
        best = fis.evaluate(0.0, 0.0, 0.0)   # short, smooth, low energy
        worst = fis.evaluate(1.0, 1.0, 1.0)  # long, brut, high energy
        mid = fis.evaluate(0.5, 0.5, 0.5)
        assert best > 0.75
        assert worst < 0.25
        assert worst < mid < best


def test_rank_routes_prefers_dominating_route():
    fis = MamdaniFIS()
    qualities = rank_routes(fis, [(1.0, 2.0, 3.0), (2.0, 4.0, 6.0)])
    assert qualities[0] > qualities[1]


def test_usv_energy_currents():
    """Moving with the current consumes less energy -- Equation (3)."""
    with_current = usv_energy(1.0, (2.0, 0.0), (1.0, 0.0), fuel_rate=10.0)
    against_current = usv_energy(1.0, (2.0, 0.0), (-1.0, 0.0), fuel_rate=10.0)
    assert with_current < against_current


def _grid_graph(size=5, seed=7):
    rng = random.Random(seed)
    g = Graph()
    for r in range(size):
        for c in range(size):
            g.add_node((r, c), float(c), float(r))
    for r in range(size):
        for c in range(size):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < size and cc < size:
                    d = g.euclidean((r, c), (rr, cc))
                    current = (rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5))
                    g.add_arc(
                        (r, c), (rr, cc),
                        energy=usv_energy(d, (3.0, 0.0), current, 10.0),
                    )
    return g


def test_sigpaf_both_variants_feasible():
    g = _grid_graph()
    start, end = (0, 0), (4, 4)
    pois = [(1, 3), (3, 1), (2, 2)]
    for variant in ("mamdani", "tsk"):
        result = sigpaf(
            g, start, end, pois,
            variant=variant, k=3, max_iterations=100, max_no_improve=30,
            rng=random.Random(42),
        )
        assert result.best_route[0] == start and result.best_route[-1] == end
        for p in pois:
            assert p in result.best_route
        for i, j in zip(result.best_route, result.best_route[1:]):
            assert g.has_arc(i, j)
        assert all(math.isfinite(v) for v in result.best_objectives)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
