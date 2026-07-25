"""SIGPA metaheuristic -- Algorithm 3 of the paper.

A population (swarm) of particles perturbs the current route: k nodes of
the current solution are selected at random; for each selected node m
the arc leaving m in the current route is excluded from the feasible
space, the prefix of the route up to m is kept, and GPA re-plans from m
to the exit, forcing the search into a different direction.  The best
particle is compared with the incumbent under the acceptance rules of
Section 4.2.1 (Rules 1-3): improvements are always kept; a non-improving
best particle may still replace the *current* solution with a
probabilistic (Gaussian) decision so the space keeps being explored from
various directions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .evaluation import RouteEvaluator
from .gpa import gpa
from .graph import Graph, Node


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class SigpaResult:
    best_route: List[Node]
    best_score: float
    initial_route: List[Node]
    initial_score: float
    iterations: int
    history: List[float] = field(default_factory=list)


def sigpa(
    graph: Graph,
    start: Node,
    end: Node,
    pois: Sequence[Node],
    k: int = 3,
    max_iterations: int = 8000,
    max_no_improve: int = 80,
    acceptance_probability: float = 0.3,
    tol: float = 0.0,
    evaluator: Optional[Callable[[Sequence[Node]], float]] = None,
    rng: Optional[random.Random] = None,
) -> SigpaResult:
    """Solve the multi-objective route-planning problem with SIGPA.

    Parameters follow Section 5.1 of the paper: the experiments used
    ``max_iterations=8000``, ``max_no_improve=80`` and
    ``acceptance_probability=0.3``.  ``k`` is the population size (number
    of nodes randomly selected from the current route per iteration).
    """
    rng = rng or random.Random()
    evaluate = evaluator or RouteEvaluator(graph)

    # Initialization stage (steps 1-4): initial solution via GPA.
    s0 = gpa(graph, start, end, pois)
    if s0 is None:
        raise RuntimeError("GPA could not construct an initial feasible route")
    e0 = evaluate(s0)
    s_current, s_best = list(s0), list(s0)
    e_current, e_best = e0, e0

    # N: route nodes of the current solution not yet selected (interior
    # nodes only -- perturbing the start or exit is not meaningful).
    unselected = set(s_current[1:-1])

    history = [e_best]
    seen_scores = [e0]  # for standardizing scores in the acceptance rule
    iteration = 0
    no_improve = 0

    while (
        len(unselected) >= k
        and iteration < max_iterations
        and no_improve < max_no_improve
    ):
        iteration += 1

        # Population stage (steps 6-23): k particles re-plan from k
        # randomly selected nodes of the current route.
        selected = rng.sample(sorted(unselected, key=repr), k)
        candidates = []
        for m in selected:
            unselected.discard(m)
            idx = s_current.index(m)
            prefix = s_current[: idx + 1]
            excluded = frozenset({(m, s_current[idx + 1])})
            pending = [p for p in pois if p not in prefix]
            prev = s_current[idx - 1] if idx > 0 else None
            tail = gpa(
                graph, m, end, pending,
                excluded_arcs=excluded, prev=prev,
            )
            if tail is None:
                continue  # infeasible particle, k_i is reduced (step 14)
            solution = prefix + tail[1:]
            candidates.append((evaluate(solution), solution))

        if not candidates:
            no_improve += 1
            history.append(e_best)
            continue

        # Evaluation stage (steps 24-34).
        e_new, s_new = min(candidates, key=lambda c: c[0])
        seen_scores.append(e_new)

        if e_new < e_best:
            # Rule 1: better than the incumbent -- accept everywhere.
            improvement = e_best - e_new
            s_best, e_best = list(s_new), e_new
            s_current, e_current = list(s_new), e_new
            unselected = set(s_current[1:-1])
            no_improve = 0
            if improvement < tol:
                history.append(e_best)
                break
        else:
            # Rules 2-3: probabilistic acceptance of the non-improving
            # best particle as the *current* solution.  The paper draws
            # the probability from the standard normal distribution,
            # r = P(e* < Z < e); scores are standardized over the scores
            # observed so far to map them onto Z.
            mu = sum(seen_scores) / len(seen_scores)
            var = sum((s - mu) ** 2 for s in seen_scores) / len(seen_scores)
            sd = math.sqrt(var)
            if sd > 0:
                r = _phi((e_new - mu) / sd) - _phi((e_best - mu) / sd)
            else:
                r = 0.0
            if r >= acceptance_probability:
                s_current, e_current = list(s_new), e_new
                unselected = set(s_current[1:-1])
            no_improve += 1

        history.append(e_best)

    return SigpaResult(
        best_route=s_best,
        best_score=e_best,
        initial_route=list(s0),
        initial_score=e0,
        iterations=iteration,
        history=history,
    )
