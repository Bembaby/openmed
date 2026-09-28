"""Exercise tokenizer offsets through the public token projection API."""

from collections import UserDict
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import pytest

from openmed.ner.adapter import to_token_classification
from openmed.ner.infer import Entity


class OffsetMapping(Mapping):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize("container", [dict, UserDict, MappingProxyType, OffsetMapping])
@pytest.mark.parametrize(
    "scheme,labels", [("BIO", ["B-TEST", "I-TEST"]), ("BILOU", ["B-TEST", "L-TEST"])]
)
def test_mapping_preserves_subword_offsets(container, scheme, labels):
    text = "synthetic"
    entity = Entity(text, 0, len(text), "TEST", 0.9)
    output = container({"offset_mapping": [(0, 3), (3, len(text))]})
    result = to_token_classification(
        [entity], text, tokenizer=lambda *a, **k: output, scheme=scheme
    )
    assert [(t.token, t.start, t.end) for t in result.tokens] == [
        ("syn", 0, 3),
        ("thetic", 3, 9),
    ]
    assert result.labels() == labels


@pytest.mark.parametrize("wrapped", [False, True])
def test_mapping_provider_and_positional_fallback(wrapped):
    def tokenizer(text):
        return UserDict({"offset_mapping": [(0, 1), (1, 3), (3, 3)]})

    supplied = (
        SimpleNamespace(get_tokenizer=lambda: tokenizer) if wrapped else tokenizer
    )
    result = to_token_classification([], "abc", tokenizer=supplied)
    assert [t.token for t in result.tokens] == ["a", "bc"]
    assert result.labels() == ["O", "O"]


@pytest.mark.parametrize(
    "output", [None, {}, UserDict(), UserDict({"offset_mapping": []})]
)
def test_missing_offsets_keep_whitespace_fallback(output):
    result = to_token_classification([], "ab cd", tokenizer=lambda *a, **k: output)
    assert [t.token for t in result.tokens] == ["ab", "cd"]


def test_no_tokenizer_keeps_existing_fallback():
    assert [t.token for t in to_token_classification([], "ab cd").tokens] == [
        "ab",
        "cd",
    ]
