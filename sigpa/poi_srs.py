"""POI Selection Ranking System (POI-SRS) -- Algorithm 1 of the paper.

Chooses the next POI to visit: POIs are ranked by ascending distance
from the current/starting point s (rank_s) and by descending distance
from the ending point e (rank_e).  The POI minimizing
rank(p) = rank_s(p) + rank_e(p) is selected, i.e. the POI close to the
start and far from the exit is visited first.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from .graph import Graph, Node


def poi_srs(
    graph: Graph,
    pois: Sequence[Node],
    start: Node,
    end: Node,
    distance: Optional[Callable[[Node, Node], float]] = None,
) -> Node:
    if not pois:
        raise ValueError("POI set is empty")
    d = distance or graph.euclidean

    by_start = sorted(pois, key=lambda p: d(p, start))
    by_end = sorted(pois, key=lambda p: d(p, end), reverse=True)
    rank_s = {p: i + 1 for i, p in enumerate(by_start)}
    rank_e = {p: i + 1 for i, p in enumerate(by_end)}
    return min(pois, key=lambda p: rank_s[p] + rank_e[p])
