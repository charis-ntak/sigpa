"""SIGPAF: SIGPA with fuzzy-logic evaluation of the objectives.

Reference implementation of:
    C. Ntakolia, D. V. Lyridis, "A Swarm Intelligence Graph-Based
    Pathfinding Algorithm Based on Fuzzy Logic (SIGPAF): A Case Study on
    Unmanned Surface Vehicle Multi-Objective Path Planning",
    J. Mar. Sci. Eng. 2021, 9(11), 1243.

The algorithm is SIGPA (Algorithm 3 of the COR 2021 paper) with one
change: the evaluation of the objectives.  A fuzzy inference system --
Mamdani (SIGPAF-M) or Takagi-Sugeno-Kang (SIGPAF-TSK) -- scores both
each candidate node during the greedy path search and the retrieved
paths of the population, over the three USV objective terms of the
paper: traveled distance, path deviations, and energy consumption.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .fuzzy import MamdaniFIS, TSKFIS, rank_routes
from .gpa import gpa
from .graph import Graph, Node


def usv_energy(
    distance: float,
    usv_velocity: Tuple[float, float],
    current_velocity: Tuple[float, float],
    fuel_rate: float,
) -> float:
    """Arc energy consumption -- Equation (3) of the paper.

    EC = d / |V + v_c| * F, where V is the USV velocity vector, v_c the
    sea-current velocity vector, and F the fuel consumption per unit
    time (kg/h).  Moving with the current increases |V + v_c| and thus
    lowers the energy needed for the arc.
    """
    vx = usv_velocity[0] + current_velocity[0]
    vy = usv_velocity[1] + current_velocity[1]
    speed = math.hypot(vx, vy)
    if speed == 0:
        return float("inf")
    return distance / speed * fuel_rate


def route_objectives(graph: Graph, route: Sequence[Node]) -> Tuple[float, float, float]:
    """Route totals (distance, deviations, energy) -- Equations (1)-(3)."""
    distance = deviation = energy = 0.0
    for idx in range(len(route) - 1):
        i, j = route[idx], route[idx + 1]
        data = graph.arc(i, j)
        distance += data.distance
        energy += data.energy
        if idx + 2 < len(route):
            deviation += graph.turn_penalty(i, j, route[idx + 2])
    return distance, deviation, energy


@dataclass
class SigpafResult:
    best_route: List[Node]
    best_objectives: Tuple[float, float, float]
    initial_route: List[Node]
    initial_objectives: Tuple[float, float, float]
    iterations: int
    history: List[Tuple[float, float, float]] = field(default_factory=list)


def sigpaf(
    graph: Graph,
    start: Node,
    end: Node,
    pois: Sequence[Node],
    variant: str = "mamdani",
    k: int = 3,
    max_iterations: int = 8000,
    max_no_improve: int = 80,
    acceptance_probability: float = 0.3,
    rng: Optional[random.Random] = None,
) -> SigpafResult:
    """Solve the multi-objective path planning problem with SIGPAF.

    ``variant`` selects the fuzzy inference system: ``"mamdani"``
    (SIGPAF-M) or ``"tsk"`` (SIGPAF-TSK).  All other parameters keep the
    semantics of :func:`sigpa.sigpa`.
    """
    if variant == "mamdani":
        fis = MamdaniFIS()
    elif variant == "tsk":
        fis = TSKFIS()
    else:
        raise ValueError(f"unknown variant {variant!r}; use 'mamdani' or 'tsk'")

    rng = rng or random.Random()

    # Initialization stage: initial solution via fuzzy-guided GPA.
    s0 = gpa(graph, start, end, pois, fis=fis)
    if s0 is None:
        raise RuntimeError("GPA could not construct an initial feasible route")
    s_current, s_best = list(s0), list(s0)
    obj_best = route_objectives(graph, s_best)

    unselected = set(s_current[1:-1])
    history = [obj_best]
    seen_scores: List[float] = []
    iteration = 0
    no_improve = 0

    def _phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    while (
        len(unselected) >= k
        and iteration < max_iterations
        and no_improve < max_no_improve
    ):
        iteration += 1

        # Population stage: identical to SIGPA, with fuzzy-guided GPA.
        selected = rng.sample(sorted(unselected, key=repr), k)
        candidates: List[List[Node]] = []
        for m in selected:
            unselected.discard(m)
            idx = s_current.index(m)
            prefix = s_current[: idx + 1]
            excluded = frozenset({(m, s_current[idx + 1])})
            pending = [p for p in pois if p not in prefix]
            prev = s_current[idx - 1] if idx > 0 else None
            tail = gpa(
                graph, m, end, pending,
                excluded_arcs=excluded, prev=prev, fis=fis,
            )
            if tail is None:
                continue
            candidates.append(prefix + tail[1:])

        if not candidates:
            no_improve += 1
            history.append(obj_best)
            continue

        # Evaluation stage: the incumbent and the candidates are ranked
        # together by the FIS over their normalized objective totals.
        pool = [s_best] + candidates
        objectives = [route_objectives(graph, r) for r in pool]
        qualities = rank_routes(fis, objectives)
        q_best = qualities[0]
        cand_qualities = qualities[1:]
        j = max(range(len(candidates)), key=lambda i: cand_qualities[i])
        q_new, s_new = cand_qualities[j], candidates[j]
        seen_scores.append(1.0 - q_new)

        if q_new > q_best:
            # Rule 1: the candidate outranks the incumbent.
            s_best, s_current = list(s_new), list(s_new)
            obj_best = route_objectives(graph, s_best)
            unselected = set(s_current[1:-1])
            no_improve = 0
        else:
            # Rules 2-3: probabilistic acceptance as current solution.
            mu = sum(seen_scores) / len(seen_scores)
            var = sum((s - mu) ** 2 for s in seen_scores) / len(seen_scores)
            sd = math.sqrt(var)
            if sd > 0:
                r = _phi(((1.0 - q_new) - mu) / sd) - _phi(((1.0 - q_best) - mu) / sd)
            else:
                r = 0.0
            if r >= acceptance_probability:
                s_current = list(s_new)
                unselected = set(s_current[1:-1])
            no_improve += 1

        history.append(obj_best)

    return SigpafResult(
        best_route=s_best,
        best_objectives=obj_best,
        initial_route=list(s0),
        initial_objectives=route_objectives(graph, s0),
        iterations=iteration,
        history=history,
    )
