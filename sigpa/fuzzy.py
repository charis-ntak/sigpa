"""Fuzzy inference machinery for SIGPAF.

Implements the fuzzy evaluation of:

    C. Ntakolia, D. V. Lyridis, "A Swarm Intelligence Graph-Based
    Pathfinding Algorithm Based on Fuzzy Logic (SIGPAF): A Case Study on
    Unmanned Surface Vehicle Multi-Objective Path Planning",
    J. Mar. Sci. Eng. 2021, 9(11), 1243.

Three normalized inputs -- traveled distance, path deviation, and energy
consumption -- are fuzzified with the membership functions of Figures
3-5 of the paper, combined through the 27 fuzzy rules of Table 1, and
mapped to a path-quality output (Figure 6, higher = better).  Both the
Mamdani (SIGPAF-M) and the zero-order Takagi-Sugeno-Kang (SIGPAF-TSK)
inference schemes are provided.  Pure stdlib, no dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

MembershipFunction = Callable[[float], float]


# -- membership functions -------------------------------------------------


def trimf(a: float, b: float, c: float) -> MembershipFunction:
    """Triangular membership function with feet a, c and peak b."""

    def mu(x: float) -> float:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)

    return mu


def trapmf(a: float, b: float, c: float, d: float) -> MembershipFunction:
    """Trapezoidal membership function with feet a, d and plateau [b, c]."""

    def mu(x: float) -> float:
        if x < a or x > d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if x < b:
            return (x - a) / (b - a) if b > a else 1.0
        return (d - x) / (d - c) if d > c else 1.0

    return mu


def centroid(mf: MembershipFunction, resolution: int = 200) -> float:
    """Numeric centroid of a membership function over [0, 1]."""
    num = den = 0.0
    for i in range(resolution + 1):
        x = i / resolution
        m = mf(x)
        num += x * m
        den += m
    return num / den if den else 0.0


# -- the SIGPAF fuzzy variables (Figures 3-6 of the paper) ----------------

# All three inputs share the same three fuzzy subsets over [0, 1].
INPUT_SETS: Dict[str, MembershipFunction] = {
    "low": trapmf(0.0, 0.0, 0.25, 0.5),
    "medium": trimf(0.25, 0.5, 0.75),
    "high": trapmf(0.5, 0.75, 1.0, 1.0),
}

# Output "optimal path" quality: five subsets with breakpoints at sixths.
_S = 1.0 / 6.0
OUTPUT_SETS: Dict[str, MembershipFunction] = {
    "very_low": trapmf(0.0, 0.0, _S, 2 * _S),
    "low": trimf(_S, 2 * _S, 3 * _S),
    "medium": trimf(2 * _S, 3 * _S, 4 * _S),
    "high": trimf(3 * _S, 4 * _S, 5 * _S),
    "very_high": trapmf(4 * _S, 5 * _S, 1.0, 1.0),
}

# The 27 fuzzy rules of Table 1: (distance, deviation, energy) -> quality.
# Input labels map linguistically as low/medium/high ==
# short/moderate/long (distance) and smooth/adequate/brut (deviation).
RULES: List[Tuple[Tuple[str, str, str], str]] = [
    (("low", "low", "low"), "very_high"),       # Rule 1
    (("low", "low", "medium"), "very_high"),    # Rule 2
    (("low", "medium", "low"), "very_high"),    # Rule 3
    (("medium", "low", "low"), "very_high"),    # Rule 4
    (("low", "low", "high"), "high"),           # Rule 5
    (("low", "medium", "medium"), "high"),      # Rule 6
    (("low", "high", "low"), "high"),           # Rule 7
    (("medium", "low", "medium"), "high"),      # Rule 8
    (("medium", "medium", "low"), "high"),      # Rule 9
    (("high", "low", "low"), "high"),           # Rule 10
    (("low", "medium", "high"), "medium"),      # Rule 11
    (("low", "high", "medium"), "medium"),      # Rule 12
    (("low", "high", "high"), "medium"),        # Rule 13
    (("medium", "low", "high"), "medium"),      # Rule 14
    (("medium", "medium", "medium"), "medium"), # Rule 15
    (("medium", "high", "low"), "medium"),      # Rule 16
    (("high", "low", "medium"), "medium"),      # Rule 17
    (("high", "low", "high"), "medium"),        # Rule 18
    (("high", "medium", "low"), "medium"),      # Rule 19
    (("high", "high", "low"), "medium"),        # Rule 20
    (("medium", "medium", "high"), "low"),      # Rule 21
    (("medium", "high", "medium"), "low"),      # Rule 22
    (("medium", "high", "high"), "low"),        # Rule 23
    (("high", "medium", "medium"), "low"),      # Rule 24
    (("high", "medium", "high"), "low"),        # Rule 25
    (("high", "high", "medium"), "low"),        # Rule 26
    (("high", "high", "high"), "very_low"),     # Rule 27
]


# -- inference ------------------------------------------------------------


def _clip(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class MamdaniFIS:
    """Mamdani inference: min AND, max aggregation, centroid defuzzification.

    Used by the SIGPAF-M variant of the paper.  Evaluations are memoized
    on inputs quantized to 3 decimals, which is well below the
    granularity of the membership functions.
    """

    resolution: int = 100

    def __post_init__(self):
        self._cache: Dict[Tuple[float, float, float], float] = {}

    def evaluate(self, distance: float, deviation: float, energy: float) -> float:
        key = (round(distance, 3), round(deviation, 3), round(energy, 3))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._evaluate(*key)
        self._cache[key] = result
        return result

    def _evaluate(self, distance: float, deviation: float, energy: float) -> float:
        inputs = (_clip(distance), _clip(deviation), _clip(energy))
        firing: Dict[str, float] = {}
        for antecedents, consequent in RULES:
            strength = min(
                INPUT_SETS[label](x) for label, x in zip(antecedents, inputs)
            )
            firing[consequent] = max(firing.get(consequent, 0.0), strength)
        num = den = 0.0
        for i in range(self.resolution + 1):
            x = i / self.resolution
            mu = max(
                (min(w, OUTPUT_SETS[label](x)) for label, w in firing.items()),
                default=0.0,
            )
            num += x * mu
            den += mu
        return num / den if den else 0.0


@dataclass
class TSKFIS:
    """Zero-order Takagi-Sugeno-Kang inference: crisp output as the
    firing-strength-weighted average of the rule consequents.

    Used by the SIGPAF-TSK variant of the paper.  The consequent
    constants default to the centroids of the Mamdani output sets so
    both controllers share the same output scale.
    """

    consequents: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.consequents is None:
            self.consequents = {
                label: centroid(mf) for label, mf in OUTPUT_SETS.items()
            }

    def evaluate(self, distance: float, deviation: float, energy: float) -> float:
        inputs = (_clip(distance), _clip(deviation), _clip(energy))
        num = den = 0.0
        for antecedents, consequent in RULES:
            strength = min(
                INPUT_SETS[label](x) for label, x in zip(antecedents, inputs)
            )
            num += strength * self.consequents[consequent]
            den += strength
        return num / den if den else 0.0


FIS = MamdaniFIS  # default alias


def rank_routes(
    fis, objective_vectors: Sequence[Sequence[float]]
) -> List[float]:
    """Quality of each route in a compared set (higher = better).

    Each vector holds the route totals (distance, deviation, energy);
    every objective is min-max normalized across the compared routes
    before fuzzification, mirroring the candidate-relative
    normalization of the original SIGPA evaluation.
    """
    if not objective_vectors:
        return []
    cols = list(zip(*objective_vectors))
    normalized_cols = []
    for col in cols:
        lo, hi = min(col), max(col)
        if hi == lo:
            normalized_cols.append([0.0] * len(col))
        else:
            normalized_cols.append([(v - lo) / (hi - lo) for v in col])
    return [
        fis.evaluate(*vals) for vals in zip(*normalized_cols)
    ]
