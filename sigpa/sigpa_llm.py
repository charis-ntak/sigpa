"""SIGPA-LLM: SIGPA with an LLM-designed evaluation criterion.

The third member of the SIGPA family:

* SIGPA   (COR 2021)  -- NRMSE evaluation of the objectives, hand-designed.
* SIGPAF  (JMSE 2021) -- fuzzy evaluation of the objectives, hand-designed.
* SIGPA-LLM           -- the evaluation of the objectives is designed
                         OFFLINE by an automated heuristic design loop in
                         which a large language model proposes candidate
                         evaluators informed by the search history
                         (FunSearch / EoH / ReEvo style).

The division of labor keeps SIGPA's real-time property intact:

  offline  --  ``sigpa_llm_train`` evolves the evaluator on benchmark
               scenarios of the target problem class, for a chosen
               deployment objective (``route_metric``).
  online   --  ``sigpa_llm`` runs standard SIGPA with the frozen, evolved
               evaluator: deterministic, auditable, and exactly as fast
               as the original NRMSE evaluation (same O(k) per candidate).

Without API credentials the design loop transparently degrades to random
mutation of the evaluator weights (self-tuning SIGPA); with credentials
(``pip install sigpa[llm]``) the candidates come from the LLM.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .graph import Graph, Node
from .sigpa import sigpa, SigpaResult
from .tune import Scenario, TuneResult, WeightedNRMSE, tune


@dataclass
class SigpaLLMModel:
    """The frozen product of the offline design phase."""

    evaluator: object                       # WeightedNRMSE or CodeEvaluator
    tune_result: Optional[TuneResult] = None
    design_history: List = field(default_factory=list)
    llm_proposals: int = 0
    fallback_proposals: int = 0

    @property
    def used_llm(self) -> bool:
        return self.llm_proposals > 0


def sigpa_llm_train(
    scenarios: Sequence[Scenario],
    route_metric: Optional[Callable] = None,
    generations: int = 10,
    offspring: int = 3,
    mode: str = "weights",
    model: Optional[str] = None,
    problem_description: str = "",
    client: object = None,
    rng: Optional[random.Random] = None,
    **sigpa_kwargs,
) -> SigpaLLMModel:
    """Offline phase: design the evaluation criterion for a problem class.

    ``scenarios`` are ``(graph, start, end, pois)`` training instances and
    ``route_metric(graph, route)`` is the deployment objective (defaults
    to the equally-weighted z1-z4 sum).  If the ``anthropic`` package and
    credentials are available, candidates are proposed by the LLM; any
    failure falls back to random weight mutation, so training always
    completes.
    """
    factory = None
    try:
        from .ahd import LLMEvaluatorFactory

        kwargs = {"mode": mode, "problem_description": problem_description}
        if model is not None:
            kwargs["model"] = model
        if client is not None:
            kwargs["client"] = client
        factory = LLMEvaluatorFactory(**kwargs)
    except Exception:
        factory = None  # no anthropic package / no credentials -> mutation

    result = tune(
        scenarios,
        generations=generations,
        offspring=offspring,
        rng=rng,
        candidate_factory=factory,
        route_metric=route_metric,
        **sigpa_kwargs,
    )
    return SigpaLLMModel(
        evaluator=result.best_evaluator,
        tune_result=result,
        design_history=list(factory.history) if factory is not None else [],
        llm_proposals=getattr(factory, "llm_proposals", 0),
        fallback_proposals=getattr(factory, "fallback_proposals", 0),
    )


def sigpa_llm(
    graph: Graph,
    start: Node,
    end: Node,
    pois: Sequence[Node],
    model: Optional[SigpaLLMModel] = None,
    **sigpa_kwargs,
) -> SigpaResult:
    """Online phase: standard SIGPA with the frozen evolved evaluator.

    ``model`` is the product of :func:`sigpa_llm_train`; without one the
    call is plain SIGPA (identity evaluator).  Runtime behavior is
    identical to :func:`sigpa.sigpa` -- the LLM is never in the loop.
    """
    evaluator = model.evaluator if model is not None else WeightedNRMSE()
    return sigpa(graph, start, end, pois,
                 arc_evaluator=evaluator, **sigpa_kwargs)
