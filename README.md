# SIGPA — Swarm Intelligence Graph-based Pathfinding Algorithm

Python reference implementation of:

> C. Ntakolia, D. K. Iakovidis, *"A swarm intelligence graph-based pathfinding
> algorithm (SIGPA) for multi-objective route planning"*, Computers &
> Operations Research 133 (2021) 105358.
> https://doi.org/10.1016/j.cor.2021.105358

and of its fuzzy variant **SIGPAF** (Mamdani and Takagi–Sugeno–Kang):

> C. Ntakolia, D. V. Lyridis, *"A Swarm Intelligence Graph-Based Pathfinding
> Algorithm Based on Fuzzy Logic (SIGPAF): A Case Study on Unmanned Surface
> Vehicle Multi-Objective Path Planning"*, J. Mar. Sci. Eng. 2021, 9(11), 1243.
> https://doi.org/10.3390/jmse9111243

Pure Python (stdlib only), no dependencies. Requires Python >= 3.8.

## Installation

From PyPI-style local install (inside the repository):

```bash
pip install .
```

or directly from GitHub once published:

```bash
pip install git+https://github.com/charis-ntak/sigpa.git
```

For development (editable install with test tooling):

```bash
pip install -e .[dev]
```

## Structure

| Module | Paper section | Contents |
|---|---|---|
| [graph.py](sigpa/graph.py) | §3.1, Def. 1–4 | Undirected graph with arc attributes: collision risk, cultural value/loss, distance; piecewise travel duration `T(i,j)`; geometric turn penalty `turn(i,j,k) = |φ|/180` |
| [evaluation.py](sigpa/evaluation.py) | §4.1.1, Def. 5; §3.2 | NRMSE arc metric; route evaluator with the objective terms z1–z4 (risk, distance, turns, multiple-crossover cultural penalty) |
| [poi_srs.py](sigpa/poi_srs.py) | §4.1.2, Alg. 1 | POI Selection Ranking System |
| [gpa.py](sigpa/gpa.py) | §4.1.3, Alg. 2 | Greedy graph-based pathfinding; fitness `f(i,j,e) = NRMSE(i,j) + d̂(j,e)` (Def. 6) |
| [sigpa.py](sigpa/sigpa.py) | §4.2, Alg. 3 | SIGPA metaheuristic: population stage, evaluation stage with acceptance Rules 1–3 |
| [astar.py](sigpa/astar.py) | §5 | Plain A* shortest-distance baseline used for comparison |
| [demo.py](sigpa/demo.py) | §5 | Random-scenario demo, installed as the `sigpa-demo` command |
| [fuzzy.py](sigpa/fuzzy.py) | JMSE §2.2.2 | Fuzzy inference machinery: membership functions of Figs. 3–6, the 27-rule base of Table 1, Mamdani and zero-order TSK controllers |
| [sigpaf.py](sigpa/sigpaf.py) | JMSE §2 | SIGPAF metaheuristic (SIGPAF-M / SIGPAF-TSK), USV objective terms (Eqs. 1–3) incl. current-based energy consumption |

## Usage

```python
import random
from sigpa import Graph, sigpa

g = Graph()
g.add_node(1, x=0.0, y=0.0)
g.add_node(2, x=1.0, y=0.5)
# ...
g.add_arc(1, 2, risk=0.2, cult=0.7, loss=0.1)   # distance defaults to Euclidean
# ...

result = sigpa(
    g, start=1, end=17, pois=[5, 10, 11],
    k=3,                        # population size (nodes perturbed per iteration)
    max_iterations=8000,        # paper, §5.1
    max_no_improve=80,          # paper, §5.1
    acceptance_probability=0.3, # paper, §5.1
    rng=random.Random(42),
)
print(result.best_route, result.best_score)
```

### SIGPAF — fuzzy variant

The only change with respect to SIGPA is the evaluation of the objectives: a
fuzzy inference system scores each candidate node during the greedy search and
ranks the retrieved paths of the population, over the three USV objective
terms (traveled distance, path deviations, energy consumption):

```python
from sigpa import sigpaf, usv_energy

# assign per-arc energy from sea currents (Equation 3 of the JMSE paper)
g.add_arc(1, 2, energy=usv_energy(
    distance=120.0, usv_velocity=(3.0, 0.0),
    current_velocity=(1.8, -0.4), fuel_rate=10.0,
))

result = sigpaf(g, start=1, end=17, pois=[5, 10, 11],
                variant="mamdani")   # SIGPAF-M; use "tsk" for SIGPAF-TSK
print(result.best_route, result.best_objectives)  # (distance, deviations, energy)
```

Run the demos / tests:

```bash
sigpa-demo 80 16 7                                # nodes, POIs, seed (after pip install)
python examples/demo_random_scenario.py 80 16 7   # equivalent, without installing
python examples/demo_sigpaf_usv.py 60 8 2021      # SIGPA vs SIGPAF-M vs SIGPAF-TSK
python -m pytest tests/                           # or: python tests/test_sigpa.py
```

The tests reproduce the numeric worked examples printed in the paper:
the NRMSE scores 0.258 / 0.345 / 0.362 and selection of arc (2,5)
(§4.1.1), and the POI-SRS ranking example (Fig. 5).

## Implementation notes (interpretation choices)

The paper leaves a few details open; the choices made here are:

1. **NRMSE formula** — the printed formula is typeset ambiguously; the
   implementation uses `NRMSE_a = (1/k)·sqrt(Σ_j d̂r²)`, which is the only
   reading that reproduces all three values of the worked example exactly.
2. **Acceptance rule Gaussian (Rules 2–3)** — the paper computes
   `r = P(e* < Z < e)` from the standard normal distribution but does not say
   how raw scores map onto Z. Here scores are standardized using the mean and
   standard deviation of all candidate scores observed so far.
3. **Greedy oscillation safeguard** — a memoryless greedy walk can oscillate
   between two nodes. GPA adds a small penalty per re-traversal of an arc
   already on the route (`revisit_penalty`, default 0.5), in the spirit of the
   multiple-crossover term o.f.4, plus a step budget after which the particle
   is declared infeasible.
4. **z4 crossover term** — implemented as `Σ cult·loss·(crossings − 1)` per
   arc, i.e. zero for arcs traversed once.
5. **Turn penalty** — computed geometrically from node coordinates as the
   deviation angle between consecutive arc direction vectors.

SIGPAF-specific choices:

6. **Membership functions** — taken exactly from Figures 3–6 of the JMSE
   paper: inputs use trapezoid (0, 0, 0.25, 0.5) / triangle (0.25, 0.5, 0.75) /
   trapezoid (0.5, 0.75, 1, 1); the five output sets have breakpoints at
   sixths (peaks 1/6 … 5/6 with shoulder trapezoids).
7. **TSK consequents** — the zero-order TSK constants are the centroids of the
   corresponding Mamdani output sets, so both controllers share one scale.
8. **In-search inputs** — during the greedy step the FIS inputs are the
   candidate's normalized distance to the current target, its turn penalty,
   and its normalized arc energy; a 0.001-weighted distance term breaks the
   ties caused by membership-function plateaus.

## Citing

If you use this implementation, please cite the papers:

```bibtex
@article{ntakolia2021sigpa,
  title   = {A swarm intelligence graph-based pathfinding algorithm ({SIGPA})
             for multi-objective route planning},
  author  = {Ntakolia, Charis and Iakovidis, Dimitris K.},
  journal = {Computers \& Operations Research},
  volume  = {133},
  pages   = {105358},
  year    = {2021},
  doi     = {10.1016/j.cor.2021.105358}
}

@article{ntakolia2021sigpaf,
  title   = {A Swarm Intelligence Graph-Based Pathfinding Algorithm Based on
             Fuzzy Logic ({SIGPAF}): A Case Study on Unmanned Surface Vehicle
             Multi-Objective Path Planning},
  author  = {Ntakolia, Charis and Lyridis, Dimitrios V.},
  journal = {Journal of Marine Science and Engineering},
  volume  = {9},
  number  = {11},
  pages   = {1243},
  year    = {2021},
  doi     = {10.3390/jmse9111243}
}
```

## License

MIT — see [LICENSE](LICENSE).
