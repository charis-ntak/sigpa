"""SIGPA: Swarm Intelligence Graph-based Pathfinding Algorithm.

Reference implementation of:
    C. Ntakolia, D. K. Iakovidis, "A swarm intelligence graph-based
    pathfinding algorithm (SIGPA) for multi-objective route planning",
    Computers & Operations Research 133 (2021) 105358.
"""

__version__ = "1.0.0"

from .graph import Graph
from .evaluation import nrmse, RouteEvaluator
from .poi_srs import poi_srs
from .gpa import gpa
from .sigpa import sigpa, SigpaResult

__all__ = [
    "Graph",
    "nrmse",
    "RouteEvaluator",
    "poi_srs",
    "gpa",
    "sigpa",
    "SigpaResult",
]
