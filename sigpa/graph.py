"""Undirected graph G = (N, A) with the arc attributes of the TRP model.

Each arc (i, j) carries the measures of Definitions 1-4 of the paper:
    risk(i, j)  in [0, 1]  -- collision risk factor (Definition 1)
    T(i, j)                -- arc travel duration, a piecewise function of
                              the collision risk (Definition 2)
    turn(i, j, k) in [0,1] -- turn penalty factor of consecutive arcs,
                              |phi| / 180 (Definition 3)
    cult(i, j), loss(i, j) -- cultural value / probability of losing
                              cultural interest (Definition 4)
    d(i, j)                -- arc length (Euclidean by default)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, Tuple

Node = Hashable
Arc = Tuple[Node, Node]


@dataclass
class ArcData:
    risk: float = 0.0
    cult: float = 0.0
    loss: float = 0.0
    distance: float = 0.0
    energy: float = 0.0  # energy consumption of the arc (SIGPAF, JMSE 2021)


@dataclass
class TravelDurationParams:
    """Parameters of the linear piecewise duration function (Definition 2)."""

    a: float = 0.33
    b: float = 0.66
    t_low: float = 1.0
    t_med: float = 2.0
    t_max: float = 3.0


class Graph:
    def __init__(self, duration_params: TravelDurationParams | None = None):
        self._coords: Dict[Node, Tuple[float, float]] = {}
        self._adj: Dict[Node, Dict[Node, ArcData]] = {}
        self.duration_params = duration_params or TravelDurationParams()

    # -- construction -----------------------------------------------------

    def add_node(self, n: Node, x: float, y: float) -> None:
        self._coords[n] = (x, y)
        self._adj.setdefault(n, {})

    def add_arc(
        self,
        i: Node,
        j: Node,
        risk: float = 0.0,
        cult: float = 0.0,
        loss: float = 0.0,
        distance: float | None = None,
        energy: float = 0.0,
    ) -> None:
        """Add an undirected arc; distance defaults to the Euclidean length."""
        if i not in self._coords or j not in self._coords:
            raise KeyError(f"both endpoints of arc ({i}, {j}) must be added first")
        if distance is None:
            distance = self.euclidean(i, j)
        data = ArcData(
            risk=risk, cult=cult, loss=loss, distance=distance, energy=energy
        )
        self._adj[i][j] = data
        self._adj[j][i] = data

    # -- accessors --------------------------------------------------------

    @property
    def nodes(self) -> Iterable[Node]:
        return self._coords.keys()

    def coords(self, n: Node) -> Tuple[float, float]:
        return self._coords[n]

    def neighbors(self, n: Node) -> Iterable[Node]:
        return self._adj[n].keys()

    def arc(self, i: Node, j: Node) -> ArcData:
        return self._adj[i][j]

    def has_arc(self, i: Node, j: Node) -> bool:
        return j in self._adj.get(i, {})

    def euclidean(self, i: Node, j: Node) -> float:
        (xi, yi), (xj, yj) = self._coords[i], self._coords[j]
        return math.hypot(xj - xi, yj - yi)

    # -- model measures ---------------------------------------------------

    def travel_duration(self, i: Node, j: Node) -> float:
        """Arc travel duration T(i, j) -- Definition 2."""
        p = self.duration_params
        risk = self.arc(i, j).risk
        if risk <= p.a:
            return p.t_low
        if risk <= p.b:
            return p.t_med
        return p.t_max

    def turn_penalty(self, i: Node, j: Node, k: Node) -> float:
        """Turn penalty factor turn(i, j, k) = |phi| / 180 -- Definition 3.

        phi is the deviation angle between the consecutive arcs (i, j) and
        (j, k): 0 when the route continues straight, 180 on a full U-turn.
        """
        (xi, yi) = self._coords[i]
        (xj, yj) = self._coords[j]
        (xk, yk) = self._coords[k]
        v1 = (xj - xi, yj - yi)
        v2 = (xk - xj, yk - yj)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_phi = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos_phi = max(-1.0, min(1.0, cos_phi))
        phi = math.degrees(math.acos(cos_phi))
        return abs(phi) / 180.0
