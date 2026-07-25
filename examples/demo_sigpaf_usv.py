"""USV path planning demo for SIGPAF (JMSE 2021, 9(11), 1243).

Builds a random maritime graph with sea currents assigned per arc, and
compares SIGPA (NRMSE evaluation) with SIGPAF-M (Mamdani) and
SIGPAF-TSK (Takagi-Sugeno-Kang) on the three objective terms of the
paper: traveled distance, path deviations, and energy consumption.

Usage:  python examples/demo_sigpaf_usv.py [nodes] [targets] [seed]
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigpa import Graph, route_objectives, sigpa, sigpaf, usv_energy

USV_VELOCITY = (3.0, 0.0)  # m/s
FUEL_RATE = 10.0           # kg/h


def maritime_graph(n_nodes, seed, radius=0.28):
    """Random geometric graph with per-arc sea currents (1.5-2.5 m/s)."""
    rng = random.Random(seed)
    g = Graph()
    for i in range(n_nodes):
        g.add_node(i, rng.random(), rng.random())
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = g.euclidean(i, j)
            if d <= radius:
                speed = rng.uniform(1.5, 2.5)
                angle = rng.uniform(0, 6.283185)
                current = (
                    speed * random.Random(seed + i + j).uniform(-1, 1),
                    speed * random.Random(seed + i * j).uniform(-1, 1),
                )
                g.add_arc(
                    i, j,
                    energy=usv_energy(d, USV_VELOCITY, current, FUEL_RATE),
                )
    for i in range(n_nodes):
        if not any(True for _ in g.neighbors(i)):
            j = min(
                (j for j in range(n_nodes) if j != i),
                key=lambda j: g.euclidean(i, j),
            )
            g.add_arc(i, j, energy=usv_energy(
                g.euclidean(i, j), USV_VELOCITY, (0.0, 0.0), FUEL_RATE))
    return g


def describe(name, graph, route, elapsed_ms):
    d, phi, ec = route_objectives(graph, route)
    print(
        f"{name:>10}: distance={d:7.3f}  deviations={phi:7.3f}  "
        f"energy={ec:7.3f}  len={len(route):3d}  time={elapsed_ms:8.2f} ms"
    )


def main():
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_targets = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2021

    rng = random.Random(seed)
    graph = maritime_graph(n_nodes, seed)
    nodes = list(graph.nodes)
    start, end = rng.sample(nodes, 2)
    targets = rng.sample([n for n in nodes if n not in (start, end)], n_targets)

    print(f"USV mission: {n_nodes} nodes, {n_targets} targets, "
          f"start={start}, end={end}\n")

    t0 = time.perf_counter()
    base = sigpa(graph, start, end, targets, rng=random.Random(seed))
    t_sigpa = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    mam = sigpaf(graph, start, end, targets, variant="mamdani",
                 rng=random.Random(seed))
    t_mam = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    tsk = sigpaf(graph, start, end, targets, variant="tsk",
                 rng=random.Random(seed))
    t_tsk = (time.perf_counter() - t0) * 1000

    describe("SIGPA", graph, base.best_route, t_sigpa)
    describe("SIGPAF-M", graph, mam.best_route, t_mam)
    describe("SIGPAF-TSK", graph, tsk.best_route, t_tsk)


if __name__ == "__main__":
    main()
