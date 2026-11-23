"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "api_mode": "chat_completion",
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_tldr_returns_response(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


def test_response_mode_maps_max_tokens(llm_params):
    from types import SimpleNamespace

    received_kwargs = {}

    def create_response(**kwargs):
        received_kwargs.update(kwargs)
        return SimpleNamespace(output_text="Summary")

    client = SimpleNamespace(
        responses=SimpleNamespace(create=create_response),
    )
    llm_params["api_mode"] = "response"
    paper = make_sample_paper()

    assert paper.generate_tldr(client, llm_params) == "Summary"
    assert received_kwargs["max_output_tokens"] == 16384
    assert "max_tokens" not in received_kwargs


def test_invalid_api_mode_falls_back_to_abstract(llm_params):
    llm_params["api_mode"] = "invalid"
    paper = make_sample_paper()

    assert paper.generate_tldr(make_stub_openai_client(), llm_params) == paper.abstract


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_affiliations_returns_parsed_list(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # No brackets even after the one retry (stub always returns the same prose) -> None.
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None


# ---------------------------------------------------------------------------
# _parse_affiliation_list (the tolerant parser) + single-quote regression
# ---------------------------------------------------------------------------


def test_parse_affiliation_list_handles_single_quotes():
    """Real-world failure 2026-11-23: local Qwen returns a Python single-quoted list,
    which json.loads rejects with 'Expecting value: line 1 column 2'. ast.literal_eval saves it."""
    from zotero_arxiv_daily.protocol import _parse_affiliation_list

    assert _parse_affiliation_list("['Tsinghua University', 'Peking University']") == [
        "Tsinghua University",
        "Peking University",
    ]
    # double-quoted JSON still works; dedup preserves order; [] is a valid (empty) parse
    assert _parse_affiliation_list('["A","B","A"]') == ["A", "B"]
    assert _parse_affiliation_list("Here you go: ['MIT']\nHope it helps") == ["MIT"]
    assert _parse_affiliation_list("[]") == []
    # negative controls: no list at all -> None (caller then retries / gives up, never crashes)
    assert _parse_affiliation_list("Tsinghua University, Peking University") is None
    assert _parse_affiliation_list("") is None
    assert _parse_affiliation_list(None) is None


def test_affiliations_single_quoted_end_to_end(llm_params):
    """A stub emitting a single-quoted list must parse on the first try (no retry needed)."""
    from types import SimpleNamespace

    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="['Stanford University', 'MIT']"))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = make_sample_paper().generate_affiliations(client, llm_params)
    assert result == ["Stanford University", "MIT"]


def test_affiliations_retry_recovers(llm_params):
    """First reply is unparseable prose; the one retry returns valid JSON and we recover it."""
    from types import SimpleNamespace

    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        content = "Sure, the affiliations are Tsinghua and Peking." if calls["n"] == 1 else '["Tsinghua University"]'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = make_sample_paper().generate_affiliations(client, llm_params)
    assert result == ["Tsinghua University"]
    assert calls["n"] == 2  # exactly one retry
