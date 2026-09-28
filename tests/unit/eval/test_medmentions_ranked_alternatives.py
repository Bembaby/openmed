"""Synthetic candidate IDs test ranking; no MedMentions/UMLS content."""

import json

import pytest

from openmed.clinical.grounding import Candidate, GroundedSpan
from openmed.eval.medmentions_linking import evaluate_medmentions_st21pv


def candidate(code, system="UMLS"):
    return Candidate(system, code, "synthetic concept", 0.8)


def evaluate(tmp_path, output, k=2):
    path = tmp_path / "synthetic.jsonl"
    path.write_text(
        json.dumps({"mention": "synthetic", "cui": "SYNTHETIC-GOLD"}) + "\n",
        encoding="utf-8",
    )
    return evaluate_medmentions_st21pv(
        path, provider=lambda text, depth: output, top_k=k
    )


def test_grounded_span_alternative_counts_for_top_k_not_top_one(tmp_path):
    span = GroundedSpan(
        "synthetic",
        0,
        9,
        candidates=(candidate("WRONG"),),
        alternatives=(candidate("SYNTHETIC-GOLD"),),
    )
    report = evaluate(tmp_path, span)
    assert report.metrics["top1_accuracy"] == 0
    assert report.metrics["top2_accuracy"] == 1
    assert report.metrics["abstention_rate"] == 0


@pytest.mark.parametrize("depth,expected", [(1, 0), (2, 0), (3, 1)])
def test_cutoff_applies_to_combined_ranked_candidates(tmp_path, depth, expected):
    span = GroundedSpan(
        "synthetic",
        0,
        9,
        candidates=(candidate("WRONG"),),
        alternatives=(candidate("OTHER"), candidate("SYNTHETIC-GOLD")),
    )
    report = evaluate(tmp_path, span, depth)
    assert report.metrics[f"top{depth}_accuracy"] == expected


def test_abstained_span_does_not_receive_credit_for_retained_alternatives(tmp_path):
    span = GroundedSpan(
        "synthetic", 0, 9, alternatives=(candidate("SYNTHETIC-GOLD"),), abstained=True
    )
    report = evaluate(tmp_path, span)
    assert report.metrics["abstention_rate"] == 1
    assert report.metrics["top2_accuracy"] == 0


@pytest.mark.parametrize("kind", ["sequence", "selected", "empty"])
def test_existing_provider_shapes_remain_supported(tmp_path, kind):
    output = [candidate("SYNTHETIC-GOLD")]
    if kind == "selected":
        output = GroundedSpan("synthetic", 0, 9, candidates=tuple(output))
    elif kind == "empty":
        output = []
    report = evaluate(tmp_path, output)
    assert report.metrics["top1_accuracy"] == (0 if kind == "empty" else 1)


def test_other_vocabularies_are_not_umls_hits(tmp_path):
    span = GroundedSpan(
        "synthetic",
        0,
        9,
        candidates=(candidate("WRONG"),),
        alternatives=(candidate("SYNTHETIC-GOLD", "RXNORM"),),
    )
    report = evaluate(tmp_path, span)
    assert report.metrics["top2_accuracy"] == 0
    assert "synthetic concept" not in json.dumps(report.metrics)


def test_unselected_alternatives_do_not_become_selected_codes(tmp_path):
    span = GroundedSpan("synthetic", 0, 9, alternatives=(candidate("SYNTHETIC-GOLD"),))
    report = evaluate(tmp_path, span)
    assert report.metrics["abstention_rate"] == 1
    assert report.metrics["top1_accuracy"] == 0
    assert report.metrics["top2_accuracy"] == 0
