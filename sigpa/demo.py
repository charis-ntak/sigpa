"""Random TRP scenario demo in the style of the paper's evaluation (Section 5).

Builds a random geometric graph G = (P, N), runs SIGPA and compares the
result with the initial GPA solution and the A* shortest-distance
baseline on the optimality criteria of Table 10 (risk, distance, turns,
multiple crossovers).

Usage:  sigpa-demo [nodes] [pois] [seed]
"""

from __future__ import annotations

import random
import sys
import time

from .astar import astar_route
from .evaluation import RouteEvaluator
from .graph import Graph
from .sigpa import sigpa


def random_geometric_graph(n_nodes: int, seed: int, radius: float = 0.28) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    pts = {i: (rng.random(), rng.random()) for i in range(n_nodes)}
    for i, (x, y) in pts.items():
        g.add_node(i, x, y)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if g.euclidean(i, j) <= radius:
                g.add_arc(
                    i, j,
                    risk=rng.random(),
                    cult=rng.random(),
                    loss=rng.random(),
                )
    # ensure every node has at least one arc
    for i in range(n_nodes):
        if not any(True for _ in g.neighbors(i)):
            j = min(
                (j for j in range(n_nodes) if j != i),
                key=lambda j: g.euclidean(i, j),
            )
            g.add_arc(i, j, risk=rng.random(), cult=rng.random(), loss=rng.random())
    return g


def describe(name, route, evaluator, elapsed_ms):
    if route is None:
        print(f"{name:>6}: no feasible route")
        return
    s = evaluator.score(route)
    crossovers = len(route) - 1 - len(
        {frozenset((i, j)) for i, j in zip(route, route[1:])}
    )
    print(
        f"{name:>6}: score={s.total:8.3f}  risk={s.z1_risk:7.3f}  "
        f"dist={s.z2_distance:6.3f}  turns={s.z3_turns:6.3f}  "
        f"cult={s.z4_cultural:6.3f}  crossovers={crossovers:2d}  "
        f"len={len(route):3d}  time={elapsed_ms:7.2f} ms"
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    n_nodes = int(argv[0]) if len(argv) > 0 else 50
    n_pois = int(argv[1]) if len(argv) > 1 else 10
    seed = int(argv[2]) if len(argv) > 2 else 2021

    rng = random.Random(seed)
    graph = random_geometric_graph(n_nodes, seed)
    nodes = list(graph.nodes)
    start, end = rng.sample(nodes, 2)
    pois = rng.sample([n for n in nodes if n not in (start, end)], n_pois)
    evaluator = RouteEvaluator(graph)

    print(f"G = ({n_pois}, {n_nodes})  start={start}  end={end}  POIs={sorted(pois)}\n")

    t0 = time.perf_counter()
    result = sigpa(graph, start, end, pois, k=3, rng=random.Random(seed))
    sigpa_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    astar = astar_route(graph, start, end, pois)
    astar_ms = (time.perf_counter() - t0) * 1000

    describe("GPA", result.initial_route, evaluator, 0.0)
    describe("SIGPA", result.best_route, evaluator, sigpa_ms)
    describe("A*", astar, evaluator, astar_ms)

    print(f"\nSIGPA iterations: {result.iterations}")
    print(f"SIGPA route: {' - '.join(map(str, result.best_route))}")


if __name__ == "__main__":
    main()
