"""Evaluation framework of SIGPA.

Implements:
  * the Normalized Root Mean Square Error (NRMSE) arc metric
    (Definition 5) used by GPA to rank the candidate arcs, and
  * the route evaluation score based on the objective-function terms
    z1-z4 of the MIQP model (Section 3.2), used by SIGPA to compare
    complete solutions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .graph import Graph, Node


def _min_max_normalize(values: Sequence[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def nrmse(performance_vectors: Sequence[Sequence[float]]) -> List[float]:
    """NRMSE score of each candidate arc (Definition 5, eqs. 2-3).

    ``performance_vectors[a]`` holds the k measure values of arc ``a``
    (e.g. [risk, duration, turn penalty, cultural loss]).  The optimal
    vector r* is the component-wise minimum over the candidates; the
    differences to r* are min-max normalized per measure across the
    candidates and each arc is scored

        NRMSE_a = (1 / k) * sqrt( sum_j  dr_hat[a][j] ** 2 )

    which reproduces the worked example of the paper (Section 4.1.1):
    scores 0.258, 0.345, 0.362 for the arcs (2,3), (2,4), (2,5).
    """
    if not performance_vectors:
        return []
    k = len(performance_vectors[0])
    r_star = [min(r[j] for r in performance_vectors) for j in range(k)]
    diffs = [[r[j] - r_star[j] for j in range(k)] for r in performance_vectors]
    normalized = list(zip(*(_min_max_normalize(col) for col in zip(*diffs))))
    return [
        math.sqrt(sum(d * d for d in row)) / k if k else 0.0
        for row in normalized
    ]


@dataclass
class RouteScore:
    """Breakdown of a route evaluation into the objective terms."""

    z1_risk: float
    z2_distance: float
    z3_turns: float
    z4_cultural: float
    total: float


class RouteEvaluator:
    """Scores a complete route (node sequence) with the terms z1-z4.

    z1 -- total collision risk of the traversed arcs        (o.f.1)
    z2 -- total traveled distance                           (o.f.2)
    z3 -- total turn penalty of consecutive arcs            (o.f.3)
    z4 -- cultural-experience penalty of multiple crossovers
          from the same arc: cult * loss * (crossings - 1)  (o.f.4)

    ``weights`` allows re-balancing of the terms (all 1.0 by default,
    as in the generic formulation).
    """

    def __init__(
        self,
        graph: Graph,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        self.graph = graph
        self.weights = weights

    def score(self, route: Sequence[Node]) -> RouteScore:
        g = self.graph
        z1 = z2 = z3 = z4 = 0.0
        crossings: Dict[frozenset, int] = {}
        for idx in range(len(route) - 1):
            i, j = route[idx], route[idx + 1]
            data = g.arc(i, j)
            z1 += data.risk
            z2 += data.distance
            if idx + 2 < len(route):
                z3 += g.turn_penalty(i, j, route[idx + 2])
            key = frozenset((i, j))
            crossings[key] = crossings.get(key, 0) + 1
        for key, count in crossings.items():
            if count > 1:
                i, j = tuple(key)
                if not g.has_arc(i, j):   # directed graphs: frozenset loses order
                    i, j = j, i
                data = g.arc(i, j)
                z4 += data.cult * data.loss * (count - 1)
        w1, w2, w3, w4 = self.weights
        total = w1 * z1 + w2 * z2 + w3 * z3 + w4 * z4
        return RouteScore(z1, z2, z3, z4, total)

    def __call__(self, route: Sequence[Node]) -> float:
        return self.score(route).total
