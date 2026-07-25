"""Offline automated design of the SIGPA evaluation criterion.

The runtime evaluation stays as light as the original SIGPA (an
NRMSE-family scalar score per candidate arc); this module moves the
design effort offline: a search loop tunes the evaluator's parameters
on a set of benchmark scenarios, and the winning evaluator is then
frozen for real-time use.

Two pieces:

* :class:`WeightedNRMSE` -- a parameterized generalization of the
  original evaluation (Definition 5 of the COR 2021 paper).  With all
  weights at 1.0 it reproduces the original criterion exactly; the
  weights re-balance the objective measures and the distance-to-target
  term without adding any runtime cost.

* :func:`tune` -- an offline (1+lambda) evolutionary loop that searches
  the weight space against benchmark scenarios.  ``candidate_factory``
  is the extension hook for automated heuristic design: any generator
  of evaluators can be plugged in -- including an LLM-driven one in the
  style of FunSearch / EoH / ReEvo -- and the loop will keep whichever
  candidate produces the best routes.  The identity candidate (the
  original SIGPA evaluation) is always evaluated first, so the tuned
  result is never worse than the default on the training scenarios.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .evaluation import RouteEvaluator, _min_max_normalize
from .sigpa import sigpa
from .graph import Graph, Node

Scenario = Tuple[Graph, Node, Node, Sequence[Node]]


@dataclass
class WeightedNRMSE:
    """Weighted NRMSE arc evaluator.

    ``measure_weights`` re-balance the k objective measures inside the
    NRMSE ([risk, duration, turn, loss] in the TRP instantiation) and
    ``distance_weight`` scales the normalized distance-to-target term of
    the fitness function (Definition 6).  All ones == original SIGPA.
    """

    measure_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    distance_weight: float = 1.0

    def __call__(
        self,
        vectors: Sequence[Sequence[float]],
        distances: Sequence[float],
    ) -> List[float]:
        if not vectors:
            return []
        k = len(vectors[0])
        w = self.measure_weights
        r_star = [min(r[j] for r in vectors) for j in range(k)]
        diffs = [[r[j] - r_star[j] for j in range(k)] for r in vectors]
        normalized = list(zip(*(_min_max_normalize(col) for col in zip(*diffs))))
        scores = [
            math.sqrt(sum(w[j] * row[j] * row[j] for j in range(k))) / k
            for row in normalized
        ]
        return [
            s + self.distance_weight * d for s, d in zip(scores, distances)
        ]


@dataclass
class TuneResult:
    best_evaluator: object  # WeightedNRMSE, or any evaluator the factory produced
    best_score: float
    default_score: float
    history: List[float] = field(default_factory=list)


def _score_evaluator(
    evaluator,
    scenarios: Sequence[Scenario],
    rng_seed: int,
    sigpa_kwargs: dict,
    route_metric: Optional[Callable[[Graph, Sequence[Node]], float]] = None,
) -> float:
    """Mean route-evaluation score of SIGPA over the scenarios."""
    total = 0.0
    for graph, start, end, pois in scenarios:
        result = sigpa(
            graph, start, end, pois,
            arc_evaluator=evaluator,
            rng=random.Random(rng_seed),
            **sigpa_kwargs,
        )
        if route_metric is None:
            total += RouteEvaluator(graph)(result.best_route)
        else:
            total += route_metric(graph, result.best_route)
    return total / len(scenarios)


def _mutate(parent: WeightedNRMSE, rng: random.Random) -> WeightedNRMSE:
    """Log-normal perturbation of the parent's weights."""
    def jitter(w: float) -> float:
        return max(1e-3, w * math.exp(rng.gauss(0.0, 0.4)))

    return WeightedNRMSE(
        measure_weights=tuple(jitter(w) for w in parent.measure_weights),
        distance_weight=jitter(parent.distance_weight),
    )


def tune(
    scenarios: Sequence[Scenario],
    generations: int = 20,
    offspring: int = 4,
    rng: Optional[random.Random] = None,
    candidate_factory: Optional[
        Callable[[WeightedNRMSE, random.Random], WeightedNRMSE]
    ] = None,
    route_metric: Optional[Callable[[Graph, Sequence[Node]], float]] = None,
    **sigpa_kwargs,
) -> TuneResult:
    """Offline (1+lambda) evolution of the evaluation criterion.

    ``candidate_factory(parent, rng)`` produces a candidate evaluator
    from the current best; it defaults to log-normal weight mutation of
    :class:`WeightedNRMSE`, and is the hook for richer generators
    (e.g. an LLM proposing evaluators).

    ``route_metric(graph, route)`` is the deployment objective being
    tuned for; it defaults to the equally-weighted z1-z4 sum.  Pass a
    domain-weighted metric (e.g. a safety-dominant one) to specialize
    the evaluator for that problem class.

    ``sigpa_kwargs`` are forwarded to :func:`sigpa.sigpa` -- use small
    budgets (e.g. ``max_iterations=50``) to keep tuning cheap.
    """
    rng = rng or random.Random()
    factory = candidate_factory or _mutate
    notify = getattr(factory, "feedback", None)
    seed = rng.randrange(2**30)

    best = WeightedNRMSE()
    best_score = _score_evaluator(best, scenarios, seed, sigpa_kwargs, route_metric)
    default_score = best_score
    history = [best_score]
    if notify is not None:
        notify(best, best_score)

    for _ in range(generations):
        for _ in range(offspring):
            candidate = factory(best, rng)
            score = _score_evaluator(
                candidate, scenarios, seed, sigpa_kwargs, route_metric
            )
            if notify is not None:
                notify(candidate, score)
            if score < best_score:
                best, best_score = candidate, score
        history.append(best_score)

    return TuneResult(
        best_evaluator=best,
        best_score=best_score,
        default_score=default_score,
        history=history,
    )
