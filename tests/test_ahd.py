"""Tests for the LLM-driven automated heuristic design factory.

A fake Claude client is injected so the tests run offline and free.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigpa import CodeEvaluator, Graph, LLMEvaluatorFactory, WeightedNRMSE, tune


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
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0) if self.replies else self.replies_default
        return _FakeResponse(reply)

    replies_default = "no idea"


class _FakeClient:
    def __init__(self, replies):
        self.messages = _FakeMessages(replies)


def test_weights_mode_parses_llm_proposal():
    client = _FakeClient([
        'Here you go: {"measure_weights": [2.0, 0.5, 1.5, 1.0], "distance_weight": 0.8}'
    ])
    factory = LLMEvaluatorFactory(client=client)
    candidate = factory(WeightedNRMSE(), random.Random(1))
    assert isinstance(candidate, WeightedNRMSE)
    assert candidate.measure_weights == (2.0, 0.5, 1.5, 1.0)
    assert candidate.distance_weight == 0.8


def test_invalid_llm_reply_falls_back_to_mutation():
    client = _FakeClient(["I cannot help with that."])
    parent = WeightedNRMSE()
    candidate = factory_candidate = LLMEvaluatorFactory(client=client)(
        parent, random.Random(2)
    )
    assert isinstance(candidate, WeightedNRMSE)
    assert candidate != parent  # mutated, not the parent itself


def test_code_mode_compiles_and_runs():
    source = (
        "```python\n"
        "def evaluate(vectors, distances):\n"
        "    k = len(vectors[0])\n"
        "    return [\n"
        "        math.sqrt(sum(d * d for d in v)) / k + dist\n"
        "        for v, dist in zip(vectors, distances)\n"
        "    ]\n"
        "```"
    )
    client = _FakeClient([source])
    factory = LLMEvaluatorFactory(client=client, mode="code")
    candidate = factory(WeightedNRMSE(), random.Random(3))
    assert isinstance(candidate, CodeEvaluator)
    fitness = candidate([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], [0.0, 1.0])
    assert fitness[0] < fitness[1]


def test_malicious_or_broken_code_rejected():
    for source in (
        "```python\nimport os\ndef evaluate(v, d):\n    return [0.0]*len(v)\n```",
        "```python\ndef evaluate(v, d):\n    return 'nope'\n```",
    ):
        client = _FakeClient([source])
        factory = LLMEvaluatorFactory(client=client, mode="code")
        candidate = factory(WeightedNRMSE(), random.Random(4))
        assert isinstance(candidate, WeightedNRMSE)  # fell back to mutation


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


def test_tune_with_llm_factory_records_feedback():
    replies = [
        '{"measure_weights": [1.5, 1.0, 1.0, 1.0], "distance_weight": 1.0}',
        '{"measure_weights": [0.5, 1.2, 0.9, 1.1], "distance_weight": 0.7}',
        '{"measure_weights": [2.0, 0.8, 1.3, 0.6], "distance_weight": 1.4}',
        '{"measure_weights": [1.1, 1.1, 0.4, 1.8], "distance_weight": 0.9}',
    ]
    client = _FakeClient(replies)
    factory = LLMEvaluatorFactory(client=client)
    g = _grid_graph()
    result = tune(
        [(g, (0, 0), (3, 3), [(1, 2)])],
        generations=2, offspring=2, rng=random.Random(9),
        candidate_factory=factory,
        k=2, max_iterations=15, max_no_improve=8,
    )
    # default + 4 candidates all reported back with their scores
    assert len(factory.history) == 5
    assert all(score is not None for _, score in factory.history)
    assert result.best_score <= result.default_score
    # the LLM saw history in later prompts
    assert "score=" in client.messages.requests[-1]["messages"][0]["content"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
