"""SIGPA: Swarm Intelligence Graph-based Pathfinding Algorithm.

Reference implementation of:
    C. Ntakolia, D. K. Iakovidis, "A swarm intelligence graph-based
    pathfinding algorithm (SIGPA) for multi-objective route planning",
    Computers & Operations Research 133 (2021) 105358.

and of its fuzzy variant SIGPAF (Mamdani and Takagi-Sugeno-Kang):
    C. Ntakolia, D. V. Lyridis, "A Swarm Intelligence Graph-Based
    Pathfinding Algorithm Based on Fuzzy Logic (SIGPAF): A Case Study on
    Unmanned Surface Vehicle Multi-Objective Path Planning",
    J. Mar. Sci. Eng. 2021, 9(11), 1243.
"""

__version__ = "1.1.0"

from .graph import Graph
from .evaluation import nrmse, RouteEvaluator
from .poi_srs import poi_srs
from .gpa import gpa
from .sigpa import sigpa, SigpaResult
from .fuzzy import MamdaniFIS, TSKFIS
from .sigpaf import sigpaf, SigpafResult, usv_energy, route_objectives

__all__ = [
    "Graph",
    "nrmse",
    "RouteEvaluator",
    "poi_srs",
    "gpa",
    "sigpa",
    "SigpaResult",
    "MamdaniFIS",
    "TSKFIS",
    "sigpaf",
    "SigpafResult",
    "usv_energy",
    "route_objectives",
]
