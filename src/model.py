"""Model providers (sync + async) with usage-token reporting.

Three implementations are bundled:

- :class:`OpenAIProvider`     — OpenAI Chat Completions (also drives any
                                OpenAI-compatible API via ``base_url``)
- :class:`AnthropicProvider`  — Anthropic Claude Messages API
- :class:`MockProvider`       — deterministic rule-based classifier; lets
                                CI run without a real API key

Every provider exposes both:

  - ``call(system_prompt, text) -> ModelOutput``   (synchronous)
  - ``await acall(system_prompt, text) -> ModelOutput``  (async)

The async variant is what ``evaluator.evaluate_async`` and the FastAPI
service use to fan-out requests in parallel under an
``asyncio.Semaphore``. The sync variant remains for the existing CLI
flow and unit tests.

Adding a fourth provider (Cohere, Mistral, Gemini, …) is a matter of
subclassing :class:`ModelProvider`, implementing both ``call`` /
``acall``, and registering it in :func:`build_provider`.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.utils import Config, extract_first_json_object

log = logging.getLogger(__name__)


# ---------- result type ----------
@dataclass(frozen=True)
class ModelOutput:
    """Normalised model reply.

    ``raw`` keeps the original text so we can debug parse failures.
    ``prompt_tokens`` / ``completion_tokens`` are reported by the provider
    when available (OpenAI usage, Anthropic usage); the mock provider and
    error fallbacks report zero.
    """

    verdict: str
    category: str
    confidence: float
    reason: str
    raw: str
    parse_ok: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------- provider interface ----------
class ModelProvider(ABC):
    """Strategy interface — every provider exposes both sync and async paths.

    Most production paths should prefer ``acall`` so the evaluator and the
    FastAPI service can fan out requests in parallel.
    """

    @abstractmethod
    def call(self, system_prompt: str, text: str) -> ModelOutput: ...

    @abstractmethod
    async def acall(self, system_prompt: str, text: str) -> ModelOutput: ...


# ---------- OpenAI ----------
class OpenAIProvider(ModelProvider):
    """Thin wrapper over OpenAI's Chat Completions API.

    Both clients are created lazily so the framework still imports cleanly
    on machines without an OPENAI_API_KEY (e.g. CI running in mock mode).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client: Any | None = None
        self._aclient: Any | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; either export the key or switch "
                "model_type to 'mock' in config.yaml."
            )
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.request_timeout_seconds,
        }
        # `base_url` lets DeepSeek, OpenRouter, Together, Ollama, etc. work
        # through this provider without a new class — they all expose the
        # OpenAI Chat Completions schema.
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return kwargs

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # imported lazily; optional dependency
        except ImportError as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "openai package is not installed. Run `pip install -r requirements.txt`."
            ) from e
        self._client = OpenAI(**self._client_kwargs())
        return self._client

    def _ensure_aclient(self) -> Any:
        if self._aclient is not None:
            return self._aclient
        try:
            from openai import AsyncOpenAI  # imported lazily; optional dependency
        except ImportError as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "openai package is not installed. Run `pip install -r requirements.txt`."
            ) from e
        self._aclient = AsyncOpenAI(**self._client_kwargs())
        return self._aclient

    def _request_kwargs(self, system_prompt: str, text: str) -> dict[str, Any]:
        return {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        }

    def call(self, system_prompt: str, text: str) -> ModelOutput:
        client = self._ensure_client()
        try:
            resp = client.chat.completions.create(**self._request_kwargs(system_prompt, text))
        except Exception as e:  # pragma: no cover - network errors aren't unit-testable
            log.warning("OpenAI call failed for text=%r: %s", text[:80], e)
            return _error_output(f"openai error: {e}")
        return _openai_resp_to_output(resp, allowed_categories=set(self.config.labels))

    async def acall(self, system_prompt: str, text: str) -> ModelOutput:
        client = self._ensure_aclient()
        try:
            resp = await client.chat.completions.create(**self._request_kwargs(system_prompt, text))
        except Exception as e:  # pragma: no cover - network errors aren't unit-testable
            log.warning("OpenAI async call failed for text=%r: %s", text[:80], e)
            return _error_output(f"openai error: {e}")
        return _openai_resp_to_output(resp, allowed_categories=set(self.config.labels))


def _openai_resp_to_output(resp: Any, *, allowed_categories: set[str]) -> ModelOutput:
    raw = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return _parse_or_default(
        raw,
        allowed_categories=allowed_categories,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


# ---------- Anthropic ----------
class AnthropicProvider(ModelProvider):
    """Anthropic Claude via the official ``anthropic`` SDK.

    The Messages API takes ``system`` as a top-level argument (not a role),
    and replies with a list of content blocks — we concatenate the text
    blocks before parsing. Claude has no first-class JSON mode, so we rely
    on the prompt to enforce the JSON schema and the same robust fallback
    parser the OpenAI provider uses.

    Requires:
      - ``anthropic`` Python package (in requirements.txt)
      - ``ANTHROPIC_API_KEY`` environment variable
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client: Any | None = None
        self._aclient: Any | None = None

    def _api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; either export the key or switch "
                "model_type to 'mock' / 'openai' in config.yaml."
            )
        return api_key

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic  # imported lazily; optional dependency
        except ImportError as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "anthropic package is not installed. Run `pip install -r requirements.txt`."
            ) from e
        self._client = Anthropic(
            api_key=self._api_key(),
            timeout=self.config.request_timeout_seconds,
        )
        return self._client

    def _ensure_aclient(self) -> Any:
        if self._aclient is not None:
            return self._aclient
        try:
            from anthropic import AsyncAnthropic  # imported lazily; optional dependency
        except ImportError as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "anthropic package is not installed. Run `pip install -r requirements.txt`."
            ) from e
        self._aclient = AsyncAnthropic(
            api_key=self._api_key(),
            timeout=self.config.request_timeout_seconds,
        )
        return self._aclient

    def _request_kwargs(self, system_prompt: str, text: str) -> dict[str, Any]:
        return {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": text}],
        }

    def call(self, system_prompt: str, text: str) -> ModelOutput:
        client = self._ensure_client()
        try:
            resp = client.messages.create(**self._request_kwargs(system_prompt, text))
        except Exception as e:  # pragma: no cover - network errors aren't unit-testable
            log.warning("Anthropic call failed for text=%r: %s", text[:80], e)
            return _error_output(f"anthropic error: {e}")
        return _anthropic_resp_to_output(resp, allowed_categories=set(self.config.labels))

    async def acall(self, system_prompt: str, text: str) -> ModelOutput:
        client = self._ensure_aclient()
        try:
            resp = await client.messages.create(**self._request_kwargs(system_prompt, text))
        except Exception as e:  # pragma: no cover - network errors aren't unit-testable
            log.warning("Anthropic async call failed for text=%r: %s", text[:80], e)
            return _error_output(f"anthropic error: {e}")
        return _anthropic_resp_to_output(resp, allowed_categories=set(self.config.labels))


def _anthropic_resp_to_output(resp: Any, *, allowed_categories: set[str]) -> ModelOutput:
    raw = _join_anthropic_text(resp)
    usage = getattr(resp, "usage", None)
    # Anthropic uses input_tokens/output_tokens; normalise to OpenAI-style
    # naming so downstream code (cost, summary) doesn't need to branch.
    return _parse_or_default(
        raw,
        allowed_categories=allowed_categories,
        prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def _join_anthropic_text(resp: Any) -> str:
    """Pull every text block out of an Anthropic Messages response."""
    blocks = getattr(resp, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        # SDK objects expose ``.type`` and ``.text``; we accept dict-shaped
        # fakes in tests too so the test suite doesn't depend on the SDK.
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type == "text":
            text = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else None
            )
            if text:
                parts.append(str(text))
    return "".join(parts)


# ---------- Mock ----------
class MockProvider(ModelProvider):
    """Deterministic rule-based classifier.

    Useful for two things:
      1. Running the full evaluation pipeline in CI without an API key.
      2. Establishing a baseline so that any LLM provider you swap in must
         beat the rules to be worth the API spend.

    Rules mirror the system prompt's decision order so that prompt and mock
    stay roughly in sync — but the LLM should still outperform on edge cases.
    """

    _URL_RE = re.compile(r"https?://|t\.me/|bit\.ly|tinyurl|wa\.me|@\w{3,}", re.IGNORECASE)
    _PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{7,}\d)")
    _PLATFORMS = re.compile(
        r"\b(insta(gram)?|инст[еау]|telegram|телеграм|тг|whats?app|вотс?ап|ватсап|"
        r"viber|signal|snap(chat)?|tiktok|дискорд|discord|skype|email|почт[ауы])\b",
        re.IGNORECASE,
    )
    _SPAM = re.compile(
        r"\b(crypto|forex|invest|earn|заработ|курс|signals|сигнал|"
        r"скидк|promo|промо|купи|купить|massage|массаж|гуарантирова?н|guarantee)\b",
        re.IGNORECASE,
    )
    _MONEY = re.compile(r"\$\d|\d+\s?(usd|eur|грн|руб|btc|eth)", re.IGNORECASE)
    _AGGRESSION = re.compile(
        r"\b(if you smiled|если бы (ты )?улыбал|для своих лет|for your age|"
        r"ну ничего|неплохо для|not bad for|пытаешься|ты бы (была|был) лучше|"
        r"you'?d (be|look) (better|cuter|prettier))",
        re.IGNORECASE,
    )

    def __init__(self, config: Config) -> None:
        self.config = config

    def call(self, system_prompt: str, text: str) -> ModelOutput:  # noqa: ARG002 - prompt unused on mock
        category, confidence, reason = self._classify(text)
        verdict = "BLOCK" if category in self.config.block_labels else "ALLOW"
        return ModelOutput(
            verdict=verdict,
            category=category,
            confidence=confidence,
            reason=reason,
            raw=f"mock:{category}",
            parse_ok=True,
        )

    async def acall(self, system_prompt: str, text: str) -> ModelOutput:
        # MockProvider has no real I/O: a sync delegation is correct and
        # keeps the rule logic in one place.
        return self.call(system_prompt, text)

    def _classify(self, text: str) -> tuple[str, float, str]:
        # Order matters: a single message can match more than one rule, so we
        # check the most "promotional" signals first. A bit.ly link inside a
        # crypto pitch is spam, not a platform switch.
        spam_hit = bool(self._SPAM.search(text) or self._MONEY.search(text))
        platform_hit = bool(
            self._PLATFORMS.search(text) or self._URL_RE.search(text) or self._PHONE_RE.search(text)
        )
        if spam_hit:
            return "spam", 0.9, "rule: promotional / monetary keyword"
        if platform_hit:
            return "gray_platform_switch", 0.85, "rule: off-platform indicator"
        if self._AGGRESSION.search(text):
            return "hidden_aggression", 0.8, "rule: backhanded / age-shaming pattern"
        return "ok", 0.7, "rule: no red flags"


# ---------- factory ----------
def build_provider(config: Config) -> ModelProvider:
    """Return the provider implementation named in ``config.model_type``.

    Add a new provider: extend the dict below and import it lazily.
    """
    kind = config.model_type.strip().lower()
    if kind == "openai":
        return OpenAIProvider(config)
    if kind == "anthropic":
        return AnthropicProvider(config)
    if kind == "mock":
        return MockProvider(config)
    raise ValueError(f"unknown model_type: {kind!r}; expected one of: openai, anthropic, mock")


# ---------- public entry point ----------
def call_model(provider: ModelProvider, system_prompt: str, text: str) -> dict[str, Any]:
    """Call the model and return a plain dict ready for serialisation.

    The dict is the public, stable shape — internals (`ModelOutput`) are an
    implementation detail.
    """
    out = provider.call(system_prompt, text)
    return {
        "verdict": out.verdict,
        "category": out.category,
        "confidence": float(out.confidence),
        "reason": out.reason,
        "parse_ok": out.parse_ok,
        "raw": out.raw,
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": out.completion_tokens,
    }


# ---------- helpers ----------
def _error_output(reason: str) -> ModelOutput:
    """The standard fail-closed fallback when the underlying API call raises."""
    return ModelOutput(
        verdict="BLOCK",
        category="other",
        confidence=0.0,
        reason=reason,
        raw="",
        parse_ok=False,
    )


def _parse_or_default(
    raw: str,
    *,
    allowed_categories: set[str],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> ModelOutput:
    """Turn a raw model reply into a normalised :class:`ModelOutput`.

    On any parse failure we log and fall back to BLOCK with confidence 0.0,
    but the usage tokens (if any) are still attributed — a parse failure
    still costs money.
    """
    obj = extract_first_json_object(raw)
    if obj is None:
        log.warning("failed to parse JSON from model reply: %r", raw[:200])
        return ModelOutput(
            verdict="BLOCK",
            category="other",
            confidence=0.0,
            reason="invalid JSON in model reply; defaulted to BLOCK",
            raw=raw,
            parse_ok=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    category = str(obj.get("category", "other"))
    if category not in allowed_categories:
        log.warning("model returned unknown category %r; coerced to 'other'", category)
        category = "other"

    verdict = str(obj.get("verdict", "BLOCK")).upper()
    if verdict not in {"ALLOW", "BLOCK"}:
        verdict = "BLOCK"

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(obj.get("reason", "")).strip() or "no reason"

    return ModelOutput(
        verdict=verdict,
        category=category,
        confidence=confidence,
        reason=reason,
        raw=raw,
        parse_ok=True,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
