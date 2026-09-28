"""Deterministic operation counts replace noisy wall-clock benchmarks."""

import pytest

from openmed.core.results import AnalyzeResult
from openmed.processing.outputs import EntityPrediction

KEYS = ("text", "entities", "model_name", "timestamp", "processing_time", "metadata")


def result():
    return AnalyzeResult(
        "synthetic",
        [EntityPrediction("synthetic", "PERSON", 0.9, 0, 9)],
        "synthetic-model",
        "synthetic-time",
        metadata={"nested": [1]},
    )


@pytest.mark.parametrize(
    "operation,expected",
    [
        (len, 6),
        (bool, True),
        (lambda r: tuple(iter(r)), KEYS),
        (lambda r: tuple(r.keys()), KEYS),
    ],
)
def test_key_operations_do_not_serialize_entities(monkeypatch, operation, expected):
    calls = []
    original = EntityPrediction.to_dict

    def observed(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(EntityPrediction, "to_dict", observed)
    assert operation(result()) == expected
    assert not calls


def test_key_operations_do_not_deepcopy_metadata(monkeypatch):
    import openmed.core.results as module

    record = result()
    calls = []
    monkeypatch.setattr(module, "deepcopy", lambda value: calls.append(value) or value)
    assert len(record) == 6
    assert tuple(iter(record)) == KEYS
    assert calls == []


def test_serialized_payload_still_copies_metadata_and_matches_keys():
    record = result()
    payload = record.to_dict()
    assert tuple(payload) == tuple(record) == KEYS
    payload["metadata"]["nested"].append(2)
    assert record.metadata == {"nested": [1]}
    assert record["model_name"] == "synthetic-model"
    assert len(payload["entities"]) == 1


def test_empty_entity_list_still_has_six_keys():
    record = AnalyzeResult("", [], "synthetic", "synthetic")
    assert len(record) == 6
    assert tuple(record) == KEYS
    with pytest.raises(KeyError):
        record["not-a-key"]
