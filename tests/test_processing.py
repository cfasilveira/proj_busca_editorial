"""Testes do módulo de Processamento."""

from __future__ import annotations

import pytest

from editorial.errors import ProcessingError
from editorial.processing import (
    TextCleaner,
    Tokenizer,
    length_stats,
    min_max_normalize,
    profile,
    token_frequencies,
    z_scores,
)


class TestCleaner:
    def test_removes_urls_and_emails(self):
        result = TextCleaner().clean("Leia em https://exemplo.com e fale com a@b.com. Tudo ok.")
        assert "https://" not in result.text
        assert "@b.com" not in result.text
        assert "Tudo ok" in result.text

    def test_none_raises_fail_first(self):
        with pytest.raises(ProcessingError):
            TextCleaner().clean(None)

    def test_cleaned_to_empty_signals_no_content(self):
        result = TextCleaner().clean("https://s.com")
        assert result.had_content is False

    def test_removes_transcript_artifacts(self):
        result = TextCleaner().clean(">> Bom dia, eh, o que houve, né? hum, vejamos.")
        assert ">>" not in result.text
        assert "eh" not in result.text
        assert "né" not in result.text
        assert "hum" not in result.text
        assert "Bom dia" in result.text
        assert "vejamos" in result.text

    def test_keeps_words_containing_hesitation_substring(self):
        result = TextCleaner().clean("humanidade e hum")
        assert result.text == "humanidade e"

    def test_artifacts_can_be_disabled(self):
        result = TextCleaner(remove_transcript_artifacts=False).clean(">> Bom dia, eh, ok.")
        assert ">>" in result.text
        assert "eh" in result.text


class TestTokenizer:
    def test_empty_text_raises_fail_first(self):
        with pytest.raises(ProcessingError, match="vazio"):
            Tokenizer().tokenize("   ")

    def test_tokenize_returns_tokens(self):
        tokens = Tokenizer().tokenize("A política econômica exige transparência.")
        assert len(tokens) >= 5
        assert all(isinstance(t, str) for t in tokens)

    def test_backend_is_strategy(self):
        assert Tokenizer().backend in {
            "internal",
            "nltk",
        } or Tokenizer().backend.startswith("spacy:")


class TestNormalizer:
    def test_length_stats(self):
        stats = length_stats([10, 20, 30])
        assert stats["mean"] == 20.0
        assert stats["count"] == 3
        assert stats["std"] > 0

    def test_length_stats_empty_raises(self):
        with pytest.raises(ProcessingError):
            length_stats([])

    def test_z_scores(self):
        scores = z_scores([1, 2, 3])
        assert abs(sum(scores)) < 1e-9

    def test_z_scores_constant(self):
        assert z_scores([5, 5, 5]) == [0.0, 0.0, 0.0]

    def test_min_max_normalize(self):
        values = min_max_normalize([0, 5, 10])
        assert values[0] == 0.0
        assert values[2] == 1.0

    def test_token_frequencies_and_profile(self):
        freqs = token_frequencies([["a", "b"], ["a"]])
        assert freqs["a"] == 2
        prof = profile([2, 1], [["a", "b"], ["a"]])
        assert "length_stats" in prof
        assert "top_tokens" in prof
