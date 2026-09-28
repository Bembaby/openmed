"""Catalog validation must report malformed shapes as gate failures."""

import json

import pytest

from openmed.core.catalog_coherence import main, manifest_label_errors


@pytest.mark.parametrize("row", [[], [1], "SYNTHETIC-PRIVATE", 42, 0.5, True, None])
def test_nonobject_row_returns_a_diagnostic(tmp_path, row):
    path = tmp_path / "models.jsonl"
    path.write_text("\n\n" + json.dumps(row) + "\n", encoding="utf-8")
    errors = manifest_label_errors(manifest_path=path)
    assert len(errors) == 1
    assert "line 3" in errors[0]
    assert "JSON object" in errors[0]
    assert "SYNTHETIC-PRIVATE" not in errors[0]


def test_cli_returns_failure_instead_of_traceback(tmp_path, capsys):
    path = tmp_path / "models.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    assert main(["--manifest", str(path)]) == 1
    captured = capsys.readouterr()
    assert "Catalog coherence check failed" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_valid_catalog_still_passes(tmp_path):
    path = tmp_path / "models.jsonl"
    path.write_text(
        json.dumps({"repo_id": "synthetic/model", "canonical_labels": ["PERSON"]})
        + "\n"
    )
    assert manifest_label_errors(manifest_path=path) == []
    assert main(["--manifest", str(path)]) == 0


@pytest.mark.parametrize(
    "payload",
    [
        "{broken\n",
        '{"canonical_labels": "PERSON"}\n',
        '{"canonical_labels": ["NO_SUCH_LABEL_314159"]}\n',
    ],
)
def test_existing_invalid_catalog_paths_still_fail(tmp_path, payload):
    path = tmp_path / "models.jsonl"
    path.write_text(payload)
    assert manifest_label_errors(manifest_path=path)
