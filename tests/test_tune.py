"""Tests for the offline tuning of the lightweight evaluation criterion."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigpa import Graph, WeightedNRMSE, sigpa, tune
from sigpa.evaluation import nrmse, _min_max_normalize


def _grid_graph(size=4, seed=1):
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


def test_identity_weights_reproduce_original_evaluation():
    """WeightedNRMSE with all-ones equals nrmse + distance, elementwise."""
    vectors = [
        [0.13, 1, 0.45, 0.87],
        [0.72, 2, 0.98, 0.12],
        [0.86, 3, 0.27, 0.36],
    ]
    distances = _min_max_normalize([3.6, 2.7, 1.8])
    original = [s + d for s, d in zip(nrmse(vectors), distances)]
    weighted = WeightedNRMSE()(vectors, distances)
    for a, b in zip(original, weighted):
        assert abs(a - b) < 1e-12


def test_identity_weights_reproduce_original_routes():
    """sigpa with the identity WeightedNRMSE returns the same route as
    sigpa with the built-in evaluation."""
    g = _grid_graph(size=5, seed=7)
    start, end = (0, 0), (4, 4)
    pois = [(1, 3), (3, 1)]
    kwargs = dict(k=3, max_iterations=60, max_no_improve=20)
    base = sigpa(g, start, end, pois, rng=random.Random(3), **kwargs)
    ident = sigpa(
        g, start, end, pois,
        rng=random.Random(3), arc_evaluator=WeightedNRMSE(), **kwargs,
    )
    assert base.best_route == ident.best_route
    assert abs(base.best_score - ident.best_score) < 1e-12


def test_tune_never_worse_than_default():
    """The tuned evaluator scores at least as well as the original on
    the training scenarios (the identity candidate is evaluated first)."""
    scenarios = []
    for seed in (1, 2):
        g = _grid_graph(size=4, seed=seed)
        scenarios.append((g, (0, 0), (3, 3), [(1, 2), (2, 1)]))
    result = tune(
        scenarios,
        generations=4, offspring=3, rng=random.Random(11),
        k=2, max_iterations=20, max_no_improve=10,
    )
    assert result.best_score <= result.default_score
    assert len(result.history) == 5
    assert isinstance(result.best_evaluator, WeightedNRMSE)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
