"""Use tiny synthetic files; never load benchmark corpus records."""

import json

import pytest

from openmed.eval.datasets.cmeee import load_cmeee


def write_source(root, name, label):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"id": label, "text": "synthetic", "spans": []}
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps([row]), encoding="utf-8")
    else:
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("requested", ["test", "dev", "unknown"])
def test_missing_requested_named_split_does_not_load_train(tmp_path, requested):
    write_source(tmp_path, "CMeEE_train.json", "train-only")
    with pytest.raises(ValueError, match="requested split"):
        load_cmeee(tmp_path, split=requested, allow_repo_path=True)


def test_missing_split_rejected_before_generic_loader(tmp_path, monkeypatch):
    import openmed.eval.datasets.cmeee as module

    write_source(tmp_path, "CMeEE_train.json", "train-only")
    calls = []
    monkeypatch.setattr(
        module, "load_multilingual_ner_benchmark", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ValueError, match="requested split"):
        load_cmeee(tmp_path, split="test", allow_repo_path=True)
    assert not calls


@pytest.mark.parametrize(
    "name", ["CMeEE_test.json", "nested/CMEEE-TEST.JSONL", "cmeee_test.ndjson"]
)
def test_available_named_split_is_selected(tmp_path, name):
    write_source(tmp_path, "CMeEE_train.json", "not-this-one")
    write_source(tmp_path, name, "selected")
    result = load_cmeee(tmp_path, split="test", allow_repo_path=True)
    assert [r.record_id for r in result.records] == ["selected"]
    assert [r.split for r in result.records] == ["test"]


@pytest.mark.parametrize("alias", ["dev", "val", "validation"])
def test_validation_aliases_are_preserved(tmp_path, alias):
    write_source(tmp_path, "CMeEE_dev.json", "selected")
    assert len(load_cmeee(tmp_path, split=alias, allow_repo_path=True).records) == 1


def test_ambiguous_matching_sources_still_raise(tmp_path):
    write_source(tmp_path, "a/CMeEE_test.json", "one")
    write_source(tmp_path, "b/CMeEE_test.json", "two")
    with pytest.raises(ValueError, match="multiple CMeEE"):
        load_cmeee(tmp_path, split="test", allow_repo_path=True)


def test_generic_directory_layout_retains_compatibility(tmp_path):
    write_source(tmp_path, "records.jsonl", "generic")
    assert len(load_cmeee(tmp_path, split="test", allow_repo_path=True).records) == 1


def test_explicit_file_remains_explicit(tmp_path):
    source = write_source(tmp_path, "CMeEE_train.json", "explicit")
    assert len(load_cmeee(source, split="custom", allow_repo_path=True).records) == 1
