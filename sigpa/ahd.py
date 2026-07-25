"""LLM-driven automated heuristic design (AHD) for the SIGPA evaluator.

Plugs into :func:`sigpa.tune.tune` as a ``candidate_factory``: instead of
random log-normal mutations, a Claude model proposes candidate evaluators
informed by the search history, in the spirit of FunSearch / EoH / ReEvo.
The LLM runs strictly OFFLINE -- whatever evaluator wins is frozen,
deterministic Python at deployment, so the runtime cost and auditability
of SIGPA are unchanged.

Two modes:

* ``mode="weights"`` (default, recommended) -- the LLM proposes new
  weight vectors for :class:`sigpa.tune.WeightedNRMSE`.  No code is
  executed; safest and cheapest.
* ``mode="code"`` -- the LLM writes the body of an evaluator function
  (FunSearch-style).  Proposed code is compiled in a restricted
  namespace and validated on a probe input before being admitted; any
  failure falls back to weight mutation.  Intended for offline research
  use -- review evolved code before deploying it.

Requires the ``anthropic`` package (``pip install sigpa[llm]``) and API
credentials, unless a compatible ``client`` is injected.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .tune import WeightedNRMSE, _mutate

DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM = """You are an expert in combinatorial optimization designing the
arc-evaluation criterion of SIGPA, a swarm-intelligence greedy graph
pathfinding metaheuristic (Ntakolia & Iakovidis, Computers & Operations
Research 133 (2021) 105358).

At every greedy step the evaluator scores candidate arcs.  Each candidate
has k=4 normalized measure differences in [0,1] (order: collision risk,
travel duration, turn penalty, cultural loss; 0 = best among candidates)
and a normalized distance-to-target in [0,1].  Lower fitness = the arc is
chosen.  The default criterion is
    fitness = sqrt(sum_j w_j * d_j^2) / k  +  w_dist * distance.

You will see the search history (candidates and their achieved route
scores; LOWER IS BETTER).  Propose ONE new candidate that you expect to
beat the best so far.  Balance exploitation of what worked with
exploration of clearly different regions."""


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _extract_code(text: str) -> Optional[str]:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "def evaluate" in text:
        return text[text.index("def evaluate"):].strip()
    return None


@dataclass
class CodeEvaluator:
    """An evaluator defined by LLM-proposed Python source.

    The source must define ``evaluate(vectors, distances)`` returning one
    fitness value (float, lower = better) per candidate.  Compiled in a
    restricted namespace (``math`` plus basic builtins, no imports, no
    I/O) and validated on a probe input before use.
    """

    source: str
    _fn: object = field(default=None, repr=False, compare=False)

    def compile(self) -> None:
        namespace = {
            "math": math,
            "min": min, "max": max, "sum": sum, "len": len,
            "abs": abs, "range": range, "sorted": sorted,
            "zip": zip, "enumerate": enumerate, "float": float,
            "list": list, "__builtins__": {},
        }
        exec(compile(self.source, "<ahd-evaluator>", "exec"), namespace)
        fn = namespace.get("evaluate")
        if not callable(fn):
            raise ValueError("source does not define evaluate()")
        self._fn = fn

    def validate(self) -> None:
        probe_vectors = [
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.5, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
        probe_distances = [0.0, 0.5, 1.0]
        result = self._fn(probe_vectors, probe_distances)
        values = list(result)
        if len(values) != 3:
            raise ValueError("wrong number of fitness values")
        for v in values:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("non-finite fitness value")

    def __call__(self, vectors, distances):
        return list(self._fn(vectors, distances))


@dataclass
class LLMEvaluatorFactory:
    """Candidate factory for :func:`sigpa.tune.tune` driven by Claude.

    Usage::

        from sigpa import tune
        from sigpa.ahd import LLMEvaluatorFactory

        factory = LLMEvaluatorFactory()          # mode="weights"
        result = tune(scenarios, candidate_factory=factory,
                      route_metric=my_metric)

    ``tune`` calls ``factory(parent, rng)`` for each candidate and
    reports the achieved score back through ``factory.feedback``, so the
    LLM sees the full search history.  On any API or parsing failure the
    factory silently falls back to random weight mutation, so a tuning
    run never dies mid-flight.
    """

    mode: str = "weights"
    model: str = DEFAULT_MODEL
    client: object = None
    problem_description: str = ""
    max_history: int = 20
    history: List[Tuple[str, Optional[float]]] = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in ("weights", "code"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.client is None:
            import anthropic  # deferred so the core package stays dependency-free

            self.client = anthropic.Anthropic()
        self._pending: Optional[str] = None

    # -- tune() protocol --------------------------------------------------

    def __call__(self, parent, rng: random.Random):
        try:
            candidate = self._propose(parent)
            if candidate is not None:
                return candidate
        except Exception:
            pass
        # Fallback: behave like the default mutation factory.
        fallback = _mutate(parent if isinstance(parent, WeightedNRMSE)
                           else WeightedNRMSE(), rng)
        self._pending = self._describe(fallback) + " (fallback mutation)"
        return fallback

    def feedback(self, candidate, score: float) -> None:
        """Called by tune() with the achieved score of the last candidate."""
        label = self._pending or self._describe(candidate)
        self._pending = None
        self.history.append((label, score))

    # -- internals --------------------------------------------------------

    def _describe(self, candidate) -> str:
        if isinstance(candidate, WeightedNRMSE):
            w = ", ".join(f"{v:.3f}" for v in candidate.measure_weights)
            return f"weights=[{w}], distance_weight={candidate.distance_weight:.3f}"
        if isinstance(candidate, CodeEvaluator):
            return f"code:\n{candidate.source}"
        return repr(candidate)

    def _history_block(self) -> str:
        recent = self.history[-self.max_history:]
        if not recent:
            return "No candidates evaluated yet."
        lines = []
        for label, score in recent:
            score_txt = f"{score:.4f}" if score is not None else "pending"
            lines.append(f"- score={score_txt}: {label}")
        return "\n".join(lines)

    def _ask(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            thinking={"type": "adaptive"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("model declined the request")
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )

    def _propose(self, parent):
        context = ""
        if self.problem_description:
            context = f"Problem class being tuned for: {self.problem_description}\n\n"

        if self.mode == "weights":
            prompt = (
                f"{context}Search history (lower score is better):\n"
                f"{self._history_block()}\n\n"
                f"Current best candidate: {self._describe(parent)}\n\n"
                "Propose ONE new candidate as JSON, and nothing else:\n"
                '{"measure_weights": [w_risk, w_duration, w_turn, w_loss], '
                '"distance_weight": w}\n'
                "All weights must be positive floats."
            )
            data = _extract_json(self._ask(prompt))
            if not data:
                return None
            weights = tuple(float(w) for w in data["measure_weights"])
            if len(weights) != 4 or any(w <= 0 for w in weights):
                return None
            distance_weight = float(data["distance_weight"])
            if distance_weight <= 0:
                return None
            candidate = WeightedNRMSE(
                measure_weights=weights, distance_weight=distance_weight
            )
        else:
            prompt = (
                f"{context}Search history (lower score is better):\n"
                f"{self._history_block()}\n\n"
                f"Current best candidate: {self._describe(parent)}\n\n"
                "Write a Python function\n"
                "    def evaluate(vectors, distances):\n"
                "that returns a list with one fitness value per candidate "
                "(lower = better).  `vectors` is a list of k=4-element lists "
                "of normalized measure differences in [0,1]; `distances` is "
                "the list of normalized distances to the target.  Only "
                "`math` and basic builtins are available -- no imports, no "
                "I/O.  Keep it O(n*k): it runs at every greedy step.  "
                "Reply with a single ```python code block and nothing else."
            )
            source = _extract_code(self._ask(prompt))
            if not source:
                return None
            candidate = CodeEvaluator(source=source)
            candidate.compile()
            candidate.validate()

        self._pending = self._describe(candidate)
        return candidate
