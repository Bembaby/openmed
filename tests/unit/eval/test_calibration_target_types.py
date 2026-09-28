"""Ensure negative outcome strings cannot silently become positives."""

import math

import pytest

from openmed.eval.calibrate import CalibrationSample


def sample(**extra):
    return CalibrationSample.from_mapping(
        {"model_id": "synthetic", "label": "PERSON", "score": 0.8, **extra}
    )


@pytest.mark.parametrize(
    "target",
    [
        "false",
        "true",
        "0",
        "1",
        "",
        "PRIVATE-SYNTHETIC",
        None,
        [],
        {},
        [1],
        2,
        -1,
        0.5,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_invalid_target_is_rejected_without_echo(target):
    with pytest.raises(ValueError, match="target must be a boolean or 0/1") as caught:
        sample(target=target)
    assert "PRIVATE-SYNTHETIC" not in str(caught.value)


@pytest.mark.parametrize(
    "target,expected",
    [(False, False), (True, True), (0, False), (1, True), (0.0, False), (1.0, True)],
)
def test_boolean_and_binary_numeric_targets_remain_supported(target, expected):
    result = sample(target=target)
    assert result.target is expected
    assert result.to_dict()["target"] is expected


@pytest.mark.parametrize("key", ["is_true", "matched"])
def test_aliases_preserve_false(key):
    assert sample(**{key: False}).target is False


@pytest.mark.parametrize("key", ["is_true", "matched"])
def test_aliases_share_the_validation(key):
    with pytest.raises(ValueError, match="target must"):
        sample(**{key: "false"})


def test_missing_target_keeps_existing_default():
    assert sample().target is True


def test_explicit_false_takes_precedence_over_aliases():
    assert sample(target=False, is_true=True, matched=True).target is False
