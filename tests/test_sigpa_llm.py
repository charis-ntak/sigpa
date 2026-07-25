"""Tests for SIGPA-LLM (offline LLM-designed evaluation + online SIGPA)."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigpa import (
    Graph,
    SigpaLLMModel,
    WeightedNRMSE,
    sigpa,
    sigpa_llm,
    sigpa_llm_train,
)

class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    stop_reason = "end_turn"

    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)

    def create(self, **kwargs):
        reply = self.replies.pop(0) if self.replies else "no idea"
        return _FakeResponse(reply)


class _FakeClient:
    def __init__(self, replies):
        self.messages = _FakeMessages(replies)


def _grid_graph(size=4, seed=1):
    rng = random.Random(seed)
    g = Graph()
    for r in range(size):
        for c in range(size):
            g.add_node((r, c), float(c), float(r))
    for r in range(size):
        for c in range(size):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < size and cc < size:
                    g.add_arc((r, c), (rr, cc), risk=rng.random(),
                              cult=rng.random(), loss=rng.random())
    return g


def _scenarios():
    return [(_grid_graph(seed=s), (0, 0), (3, 3), [(1, 2), (2, 1)])
            for s in (1, 2)]


def test_train_with_llm_client():
    replies = ['{"measure_weights": [2.0, 1.0, 0.5, 1.5], "distance_weight": 0.9}'] * 6
    model = sigpa_llm_train(
        _scenarios(), generations=2, offspring=2,
        client=_FakeClient(replies), rng=random.Random(4),
        k=2, max_iterations=15, max_no_improve=8,
    )
    assert isinstance(model, SigpaLLMModel)
    assert model.tune_result.best_score <= model.tune_result.default_score
    assert len(model.design_history) >= 3  # default + candidates reported


def test_train_without_llm_falls_back_to_mutation():
    # no client and no credentials -> factory unavailable -> pure mutation
    model = sigpa_llm_train(
        _scenarios(), generations=2, offspring=2,
        client=None, rng=random.Random(4),
        k=2, max_iterations=15, max_no_improve=8,
    )
    assert isinstance(model.evaluator, WeightedNRMSE)
    assert model.tune_result.best_score <= model.tune_result.default_score


def test_online_phase_runs_frozen_evaluator():
    g = _grid_graph(seed=7)
    start, end, pois = (0, 0), (3, 3), [(1, 2)]
    model = SigpaLLMModel(evaluator=WeightedNRMSE(
        measure_weights=(2.0, 1.0, 0.5, 1.5), distance_weight=0.9))
    result = sigpa_llm(g, start, end, pois, model=model,
                       max_iterations=40, rng=random.Random(1))
    assert result.best_route[0] == start and result.best_route[-1] == end
    for p in pois:
        assert p in result.best_route
    # identity model == plain SIGPA
    plain = sigpa(g, start, end, pois, max_iterations=40, rng=random.Random(1))
    ident = sigpa_llm(g, start, end, pois, model=None,
                      max_iterations=40, rng=random.Random(1))
    assert plain.best_route == ident.best_route


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
