"""Provider-level tests with mocked SDK clients.

These tests exercise the provider plumbing (factory, response parsing,
fallbacks) without hitting any real API.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.model import (
    AnthropicProvider,
    MockProvider,
    OpenAIProvider,
    _join_anthropic_text,
    build_provider,
)
from src.utils import Config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _config(model_type: str = "mock", **overrides: Any) -> Config:
    cfg = Config.from_file(REPO_ROOT / "config.yaml")
    return replace(cfg, model_type=model_type, **overrides)


# ---------- factory ----------
def test_factory_dispatches_to_each_provider() -> None:
    assert isinstance(build_provider(_config("mock")), MockProvider)
    assert isinstance(build_provider(_config("openai")), OpenAIProvider)
    assert isinstance(build_provider(_config("anthropic")), AnthropicProvider)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown model_type"):
        build_provider(_config("gemini"))


# ---------- Anthropic ----------
class _FakeBlock:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, blocks: list[_FakeBlock]) -> None:
        self.content = blocks


class _FakeAnthropicMessages:
    def __init__(self, response: _FakeAnthropicResponse) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: _FakeAnthropicResponse) -> None:
        self.messages = _FakeAnthropicMessages(response)


def test_anthropic_provider_parses_text_block() -> None:
    raw_json = (
        '{"verdict":"BLOCK","reason":"off-platform redirect",'
        '"category":"gray_platform_switch","confidence":0.92}'
    )
    response = _FakeAnthropicResponse([_FakeBlock("text", raw_json)])
    client = _FakeAnthropicClient(response)

    provider = AnthropicProvider(_config("anthropic"))
    provider._client = client  # bypass the lazy SDK import

    out = provider.call("system", "kick it to whatsapp")
    assert out.verdict == "BLOCK"
    assert out.category == "gray_platform_switch"
    assert out.confidence == pytest.approx(0.92)
    assert out.parse_ok is True
    # The provider must pass system at the top level (not as a message).
    assert client.messages.last_kwargs is not None
    assert client.messages.last_kwargs["system"] == "system"
    assert client.messages.last_kwargs["messages"] == [
        {"role": "user", "content": "kick it to whatsapp"}
    ]


def test_anthropic_provider_falls_back_on_garbled_reply() -> None:
    # Claude returns chatty prose with no JSON — provider must default to BLOCK.
    response = _FakeAnthropicResponse([_FakeBlock("text", "Sure, I think this is fine.")])
    provider = AnthropicProvider(_config("anthropic"))
    provider._client = _FakeAnthropicClient(response)

    out = provider.call("system", "innocent message")
    assert out.parse_ok is False
    assert out.verdict == "BLOCK"
    assert out.category == "other"


def test_anthropic_provider_recovers_from_markdown_fences() -> None:
    raw = '```json\n{"verdict":"ALLOW","category":"ok","confidence":0.8}\n```'
    response = _FakeAnthropicResponse([_FakeBlock("text", raw)])
    provider = AnthropicProvider(_config("anthropic"))
    provider._client = _FakeAnthropicClient(response)

    out = provider.call("system", "hello")
    assert out.verdict == "ALLOW"
    assert out.category == "ok"


class _FakeUsage:
    """Anthropic-shaped usage object: input_tokens / output_tokens."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def test_anthropic_provider_propagates_usage_tokens() -> None:
    """input_tokens / output_tokens must be normalised to prompt_/completion_."""
    response = _FakeAnthropicResponse([_FakeBlock("text", '{"verdict":"ALLOW","category":"ok"}')])
    response.usage = _FakeUsage(input_tokens=123, output_tokens=45)  # type: ignore[attr-defined]
    provider = AnthropicProvider(_config("anthropic"))
    provider._client = _FakeAnthropicClient(response)

    out = provider.call("system", "hi")
    assert out.prompt_tokens == 123
    assert out.completion_tokens == 45


def test_anthropic_provider_handles_missing_usage_gracefully() -> None:
    """If the SDK omits usage, we report zero — not crash."""
    response = _FakeAnthropicResponse([_FakeBlock("text", '{"verdict":"ALLOW","category":"ok"}')])
    provider = AnthropicProvider(_config("anthropic"))
    provider._client = _FakeAnthropicClient(response)

    out = provider.call("system", "hi")
    assert out.prompt_tokens == 0
    assert out.completion_tokens == 0


def test_join_anthropic_text_skips_non_text_blocks() -> None:
    blocks = [
        _FakeBlock("tool_use", "ignored"),
        _FakeBlock("text", "first "),
        _FakeBlock("text", "second"),
    ]
    response = _FakeAnthropicResponse(blocks)
    assert _join_anthropic_text(response) == "first second"


def test_join_anthropic_text_handles_dict_blocks() -> None:
    """Same path, but with dict-shaped blocks (older SDK or test fakes)."""

    class _R:
        content = [{"type": "text", "text": "ok"}]

    assert _join_anthropic_text(_R()) == "ok"


# ---------- OpenAI base_url plumbing ----------
def test_openai_provider_passes_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """`base_url` from config must be forwarded to the OpenAI SDK."""
    captured: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    # Inject a fake `openai` module so the lazy import inside the provider
    # picks up our stub instead of hitting the real SDK.
    import sys
    import types

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = OpenAIProvider(_config("openai", base_url="https://api.deepseek.com/v1"))
    provider._ensure_client()

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.deepseek.com/v1"


def test_openai_provider_omits_base_url_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    import sys
    import types

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = OpenAIProvider(_config("openai", base_url=""))
    provider._ensure_client()

    assert "base_url" not in captured


# ---------- MockProvider classification ----------
@pytest.mark.parametrize(
    ("text", "expected_category", "expected_confidence", "expected_verdict"),
    [
        # Strong signals → high confidence (auto-decided, no REVIEW at default 0.5).
        ("Hey, how was your weekend?", "ok", 0.7, "ALLOW"),
        ("Купи курс по форексу за 99$, скидка 70%!", "spam", 0.9, "BLOCK"),
        ("Hi, message me on whatsapp +380501234567", "gray_platform_switch", 0.85, "BLOCK"),
        ("If you smiled more in your photos, you'd be cuter", "hidden_aggression", 0.8, "BLOCK"),
        # Weak / borderline signals → low confidence (caller routes to REVIEW).
        ("Hey, I have a special offer just for you", "spam", 0.45, "BLOCK"),
        ("Кинь номер свой если хочешь продолжить общение", "gray_platform_switch", 0.45, "BLOCK"),
        (
            "Hopefully your personality is as nice as your photos",
            "hidden_aggression",
            0.45,
            "BLOCK",
        ),
    ],
)
def test_mock_provider_classifies_strong_and_weak_signals(
    text: str,
    expected_category: str,
    expected_confidence: float,
    expected_verdict: str,
) -> None:
    """MockProvider must distinguish strong-signal patterns (high confidence,
    auto-decided) from weak-signal patterns (low confidence, REVIEW-bound)."""
    provider = MockProvider(_config("mock"))
    out = provider.call(system_prompt="ignored", text=text)
    assert out.category == expected_category
    assert out.confidence == pytest.approx(expected_confidence)
    assert out.verdict == expected_verdict
    assert out.parse_ok is True


def test_mock_provider_strong_spam_overrides_platform_signal() -> None:
    """A spam keyword in a message that ALSO contains a platform handle must
    be classified as spam (rule order: spam first)."""
    provider = MockProvider(_config("mock"))
    out = provider.call(
        system_prompt="ignored",
        text="Заработай 5000$ в месяц! Подробности в telegram @money_bot",
    )
    assert out.category == "spam"
    assert out.confidence == pytest.approx(0.9)
