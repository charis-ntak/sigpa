"""Greedy Graph-based Pathfinding Algorithm (GPA) -- Algorithm 2.

A modified A*-like best-first search that keeps only the best proceeding
node at the current position (no open/closed lists).  At every step the
fitness of each neighbor arc is (Definition 6):

    f(i, j, e) = NRMSE(i, j) + d_hat(j, e)

where NRMSE ranks the arc against its sibling candidates over the model
measures [risk, travel duration, turn penalty, cultural loss] and
d_hat(j, e) is the min-max-normalized Euclidean distance of the
candidate node j to the current target node e.  The arc with the lowest
fitness is appended to the route.

Targets are visited in the order produced by POI-SRS; once all selected
POIs are on the route the exit node becomes the final target.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence, Set, Tuple

from .evaluation import _min_max_normalize, nrmse
from .graph import Graph, Node
from .poi_srs import poi_srs

DirectedArc = Tuple[Node, Node]


def _step(
    graph: Graph,
    current: Node,
    prev: Optional[Node],
    target: Node,
    excluded: FrozenSet[DirectedArc],
    traversals: dict,
    revisit_penalty: float,
    fis=None,
    arc_evaluator=None,
) -> Optional[Node]:
    """Choose the next node from ``current`` -- steps 6-11 of Algorithm 2.

    With ``fis`` set (SIGPAF, JMSE 2021), each candidate arc is scored by
    the fuzzy inference system over the normalized distance to the
    target, the turn penalty, and the normalized arc energy; the crisp
    path-quality output (higher = better) replaces the NRMSE criterion.

    With ``arc_evaluator`` set, the callable receives the candidates'
    measure vectors and normalized distances and returns their fitness
    values directly (e.g. :class:`sigpa.tune.WeightedNRMSE`).
    """
    candidates = [
        n for n in graph.neighbors(current) if (current, n) not in excluded
    ]
    if not candidates:
        return None

    distances = _min_max_normalize(
        [graph.euclidean(n, target) for n in candidates]
    )
    turns = [
        graph.turn_penalty(prev, current, n) if prev is not None else 0.0
        for n in candidates
    ]

    if fis is not None:
        energies = _min_max_normalize(
            [graph.arc(current, n).energy for n in candidates]
        )
        # Distance participates through the FIS; keep only an epsilon
        # share to break the ties caused by membership-function plateaus
        # in favour of the candidate closest to the target.
        fitness = [
            1.0 - fis.evaluate(d, t, e) + 0.001 * d
            for d, t, e in zip(distances, turns, energies)
        ]
    else:
        vectors = []
        for n, turn in zip(candidates, turns):
            data = graph.arc(current, n)
            vectors.append(
                [data.risk, graph.travel_duration(current, n), turn, data.loss]
            )
        if arc_evaluator is not None:
            fitness = arc_evaluator(vectors, distances)
        else:
            fitness = [
                s + d for s, d in zip(nrmse(vectors), distances)
            ]

    best, best_f = None, None
    for n, f0 in zip(candidates, fitness):
        # Safeguard against greedy oscillation: re-traversing an arc already
        # on the route is increasingly discouraged (in the spirit of the
        # multiple-crossover penalty of o.f.4).
        f = f0 + revisit_penalty * traversals.get(frozenset((current, n)), 0)
        if best_f is None or f < best_f:
            best, best_f = n, f
    return best


def gpa(
    graph: Graph,
    start: Node,
    end: Node,
    pois: Sequence[Node],
    excluded_arcs: FrozenSet[DirectedArc] = frozenset(),
    prev: Optional[Node] = None,
    max_steps: Optional[int] = None,
    revisit_penalty: float = 0.5,
    fis=None,
    arc_evaluator=None,
) -> Optional[List[Node]]:
    """Construct a route from ``start`` to ``end`` visiting all ``pois``.

    Returns the node sequence, or ``None`` when no feasible route was
    found (dead end caused by the excluded arcs, or step budget hit).
    ``prev`` optionally provides the node preceding ``start`` so the turn
    penalty stays continuous when GPA extends a partial route (as done by
    the SIGPA population stage).
    """
    if max_steps is None:
        max_steps = 20 * max(len(list(graph.nodes)), 1)

    route: List[Node] = [start]
    pending: List[Node] = [p for p in pois if p != start]
    traversals: dict = {}
    current, previous = start, prev
    steps = 0

    while True:
        if pending:
            target = poi_srs(graph, pending, current, end)
        else:
            target = end

        while current != target:
            nxt = _step(
                graph, current, previous, target,
                excluded_arcs, traversals, revisit_penalty, fis,
                arc_evaluator,
            )
            if nxt is None:
                return None
            steps += 1
            if steps > max_steps:
                return None
            key = frozenset((current, nxt))
            traversals[key] = traversals.get(key, 0) + 1
            route.append(nxt)
            previous, current = current, nxt
            if nxt in pending:
                pending.remove(nxt)

        if target in pending:
            pending.remove(target)
        if not pending and current == end:
            return route
