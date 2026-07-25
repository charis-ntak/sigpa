"""LLM-driven automated heuristic design demo.

Evolves the SIGPA evaluation criterion with Claude proposing candidates
(informed by the search history) instead of random mutation, tuned for a
safety-dominant deployment objective.

Requires API credentials (`pip install anthropic` and ANTHROPIC_API_KEY,
or `ant auth login`).  Costs a few LLM calls per generation.

Usage:  python examples/demo_llm_ahd.py [mode]     # mode: weights | code
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigpa import Graph, LLMEvaluatorFactory, tune
from sigpa.evaluation import RouteEvaluator


def scenario(seed, n=50, p=8, radius=0.28):
    rng = random.Random(seed)
    g = Graph()
    for i in range(n):
        g.add_node(i, rng.random(), rng.random())
    for i in range(n):
        for j in range(i + 1, n):
            if g.euclidean(i, j) <= radius:
                g.add_arc(i, j, risk=rng.random(), cult=rng.random(),
                          loss=rng.random())
    for i in range(n):
        if not any(True for _ in g.neighbors(i)):
            j = min((j for j in range(n) if j != i),
                    key=lambda j: g.euclidean(i, j))
            g.add_arc(i, j, risk=rng.random(), cult=rng.random(),
                      loss=rng.random())
    nodes = list(g.nodes)
    s, e = rng.sample(nodes, 2)
    pois = rng.sample([x for x in nodes if x not in (s, e)], p)
    return (g, s, e, pois)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "weights"
    train = [scenario(s) for s in (1, 2, 3, 4)]
    safety = lambda g, route: RouteEvaluator(g, weights=(5, 1, 1, 1))(route)

    factory = LLMEvaluatorFactory(
        mode=mode,
        problem_description=(
            "Random geometric graphs (~50 nodes), 8 POIs to visit, "
            "safety-dominant objective: route collision risk is weighted "
            "5x relative to distance, turns and cultural loss."
        ),
    )
    result = tune(
        train,
        generations=6, offspring=3, rng=random.Random(5),
        candidate_factory=factory, route_metric=safety,
        k=3, max_iterations=60, max_no_improve=25,
    )

    print(f"default score: {result.default_score:.3f}")
    print(f"evolved score: {result.best_score:.3f}")
    print(f"best evaluator: {result.best_evaluator}")
    print("\nsearch history the LLM saw:")
    for label, score in factory.history:
        print(f"  score={score:.3f}  {label[:100]}")


if __name__ == "__main__":
    main()
