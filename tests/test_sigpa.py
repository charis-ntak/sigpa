"""Tests anchored to the numeric examples printed in the paper."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigpa import Graph, gpa, nrmse, poi_srs, sigpa
from sigpa.evaluation import RouteEvaluator, _min_max_normalize


def test_nrmse_paper_example():
    """Worked example of Section 4.1.1 (arcs (2,3), (2,4), (2,5))."""
    vectors = [
        [0.13, 1, 0.45, 0.87],  # arc (2, 3)
        [0.72, 2, 0.98, 0.12],  # arc (2, 4)
        [0.86, 3, 0.27, 0.36],  # arc (2, 5)
    ]
    scores = nrmse(vectors)
    assert math.isclose(scores[0], 0.258, abs_tol=1e-3)
    assert math.isclose(scores[1], 0.345, abs_tol=1e-3)
    assert math.isclose(scores[2], 0.362, abs_tol=1e-3)


def test_fitness_selects_arc_2_5():
    """With the paper's normalized distances (1, 0.5, 0) the arc (2,5)
    has the minimum fitness and is inserted in the solution (Step 4)."""
    scores = nrmse(
        [[0.13, 1, 0.45, 0.87], [0.72, 2, 0.98, 0.12], [0.86, 3, 0.27, 0.36]]
    )
    distances = _min_max_normalize([3.6, 2.7, 1.8])
    fitness = [s + d for s, d in zip(scores, distances)]
    assert fitness.index(min(fitness)) == 2


def test_poi_srs_paper_example():
    """POI-SRS example (Fig. 5): d(p,s) = 3, 5.7, 4.5, 6.8 and
    d(p,e) = 8.1, 4.5, 6.7, 5.2 => p1 selected (final rank 2)."""
    g = Graph()
    g.add_node("s", 0, 0)
    g.add_node("e", 10, 0)
    d_s = {"p1": 3.0, "p2": 5.7, "p3": 4.5, "p4": 6.8}
    d_e = {"p1": 8.1, "p2": 4.5, "p3": 6.7, "p4": 5.2}
    for p in d_s:
        g.add_node(p, 0, 0)

    def dist(a, b):
        if b == "s":
            return d_s[a]
        if b == "e":
            return d_e[a]
        raise ValueError

    chosen = poi_srs(g, list(d_s), "s", "e", distance=dist)
    assert chosen == "p1"


def _grid_graph(size=4, seed=1):
    """Random-weight grid graph for end-to-end runs."""
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
                    g.add_arc(
                        (r, c), (rr, cc),
                        risk=rng.random(),
                        cult=rng.random(),
                        loss=rng.random(),
                    )
    return g


def test_gpa_visits_all_pois():
    g = _grid_graph()
    start, end = (0, 0), (3, 3)
    pois = [(2, 1), (0, 3)]
    route = gpa(g, start, end, pois)
    assert route is not None
    assert route[0] == start and route[-1] == end
    for p in pois:
        assert p in route  # feasibility condition (Section 3.3)
    for i, j in zip(route, route[1:]):
        assert g.has_arc(i, j)


def test_sigpa_never_worse_than_initial():
    g = _grid_graph(size=5, seed=7)
    start, end = (0, 0), (4, 4)
    pois = [(1, 3), (3, 1), (2, 2)]
    result = sigpa(
        g, start, end, pois,
        k=3, max_iterations=200, max_no_improve=40,
        rng=random.Random(42),
    )
    assert result.best_score <= result.initial_score
    assert result.best_route[0] == start and result.best_route[-1] == end
    for p in pois:
        assert p in result.best_route
    evaluator = RouteEvaluator(g)
    assert math.isclose(evaluator(result.best_route), result.best_score)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
