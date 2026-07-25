"""Plain A* baseline (shortest distance), used for comparison in the paper.

Visits the selected POIs in POI-SRS order, planning each leg with the
classic A* over arc distances with the Euclidean heuristic.
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import List, Optional, Sequence

from .graph import Graph, Node
from .poi_srs import poi_srs


def astar_leg(graph: Graph, start: Node, goal: Node) -> Optional[List[Node]]:
    tie = count()
    open_heap = [(graph.euclidean(start, goal), next(tie), start)]
    g_cost = {start: 0.0}
    came_from = {}
    closed = set()
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        if current in closed:
            continue
        closed.add(current)
        for n in graph.neighbors(current):
            tentative = g_cost[current] + graph.arc(current, n).distance
            if tentative < g_cost.get(n, float("inf")):
                g_cost[n] = tentative
                came_from[n] = current
                heapq.heappush(
                    open_heap,
                    (tentative + graph.euclidean(n, goal), next(tie), n),
                )
    return None


def astar_route(
    graph: Graph, start: Node, end: Node, pois: Sequence[Node]
) -> Optional[List[Node]]:
    """Full A* tour: start -> POIs (in POI-SRS order) -> end."""
    route: List[Node] = [start]
    pending = [p for p in pois if p != start]
    current = start
    while pending:
        target = poi_srs(graph, pending, current, end)
        pending.remove(target)
        if target in route:
            continue
        leg = astar_leg(graph, current, target)
        if leg is None:
            return None
        route.extend(leg[1:])
        current = target
    leg = astar_leg(graph, current, end)
    if leg is None:
        return None
    route.extend(leg[1:])
    return route
