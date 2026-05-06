"""FastAPI service exposing the moderator as an HTTP endpoint.

The service loads the same ``config.yaml`` / ``system_prompt.txt`` /
``ModelProvider`` stack the CLI uses, but **once at startup** — not per
request — so each request is just one provider call plus JSON parsing.

Endpoints
---------
GET  /               — service info + link to /docs
GET  /health         — liveness/readiness probe (does not call the model)
POST /moderate       — moderate one message
POST /moderate/batch — moderate up to ``MAX_BATCH_SIZE`` messages

Auth
----
If the ``API_AUTH_TOKEN`` environment variable is set, every request must
carry ``Authorization: Bearer <token>`` matching it. Leave it unset for
local development — the service will accept all requests.

Run
---
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    # or:
    python -m src.api  --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Final

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.cost import estimate_cost
from src.model import ModelOutput, ModelProvider, build_provider
from src.utils import Config, configure_logging

log = logging.getLogger("moderator.api")

MAX_BATCH_SIZE: Final[int] = 64
MAX_TEXT_LENGTH: Final[int] = 8000


# ---------- Pydantic schemas ----------
class ModerateRequest(BaseModel):
    """Input for ``POST /moderate``."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Message body to classify.",
        examples=["Привет! Куда ездила в отпуск?"],
    )


class BatchModerateRequest(BaseModel):
    """Input for ``POST /moderate/batch``.

    Per-item ``min_length`` / ``max_length`` constraints mirror the single
    endpoint so a malformed item in a batch fails fast with 422 instead of
    silently hitting the LLM with empty / oversized strings.
    """

    texts: list[Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH)]] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Up to {MAX_BATCH_SIZE} messages to classify in one call.",
    )


class UsageInfo(BaseModel):
    """Token counts + estimated USD cost reported by the provider."""

    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)
    estimated_cost_usd: float | None = Field(
        None,
        description=(
            "Estimated USD cost for this call. Null when the model isn't in the price table."
        ),
    )


class ModerateResponse(BaseModel):
    """Normalised model reply, returned by both endpoints."""

    verdict: str = Field(..., description="``ALLOW`` or ``BLOCK``.")
    category: str = Field(..., description="One of the labels in config.yaml.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Short human-readable rationale.")
    parse_ok: bool = Field(
        ...,
        description="False when the model's reply was not valid JSON and we fell back to BLOCK.",
    )
    raw: str = Field(
        "",
        description="Raw model reply (kept for debugging; empty in many fallback paths).",
    )
    usage: UsageInfo = Field(default_factory=UsageInfo)


class BatchModerateResponse(BaseModel):
    results: list[ModerateResponse]
    usage: UsageInfo = Field(
        default_factory=UsageInfo,
        description="Aggregated tokens + cost across every item in the batch.",
    )


class HealthResponse(BaseModel):
    status: str = Field("ok")
    model_type: str
    model_name: str
    base_url: str = Field("", description="Set when using an OpenAI-compatible endpoint.")
    auth_required: bool


class ServiceInfo(BaseModel):
    service: str = "strict-moderator"
    version: str = "0.1.0"
    docs_url: str = "/docs"
    health_url: str = "/health"


# ---------- application state ----------
class _AppState:
    """Holds singletons created during the FastAPI lifespan."""

    config: Config | None = None
    provider: ModelProvider | None = None
    system_prompt: str = ""


_state = _AppState()


def _build_state(config_path: Path | str) -> None:
    """Initialise ``_state`` from disk. Idempotent."""
    config = Config.from_file(config_path)
    configure_logging(config.log_level)
    system_prompt = config.prompt_path.read_text(encoding="utf-8")
    provider = build_provider(config)
    _state.config = config
    _state.provider = provider
    _state.system_prompt = system_prompt
    log.info(
        "api ready: model_type=%s model_name=%s auth=%s",
        config.model_type,
        config.model_name,
        "on" if os.getenv("API_AUTH_TOKEN") else "off",
    )


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    config_path = os.getenv("MODERATOR_CONFIG", "config.yaml")
    _build_state(config_path)
    yield


# ---------- auth dependency ----------
def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    """Bearer-token auth — only enforced if ``API_AUTH_TOKEN`` is set."""
    expected = os.getenv("API_AUTH_TOKEN", "").strip()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = authorization.removeprefix("Bearer ").strip()
    if presented != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------- core helpers ----------
def _model_name() -> str:
    return _state.config.model_name if _state.config is not None else ""


def _usage_for(prompt_tokens: int, completion_tokens: int) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=estimate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=_model_name(),
        ),
    )


def _to_response(out: ModelOutput) -> ModerateResponse:
    return ModerateResponse(
        verdict=out.verdict,
        category=out.category,
        confidence=out.confidence,
        reason=out.reason,
        parse_ok=out.parse_ok,
        raw=out.raw,
        usage=_usage_for(out.prompt_tokens, out.completion_tokens),
    )


def _provider() -> ModelProvider:
    if _state.provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="moderator is not initialised yet",
        )
    return _state.provider


# ---------- FastAPI app ----------
app = FastAPI(
    title="strict-moderator",
    description=(
        "HTTP wrapper around the strict-moderator evaluation framework. "
        "Pick a provider in config.yaml, set the matching API key as an "
        "environment variable, run `uvicorn src.api:app`, and POST messages "
        "to `/moderate`."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


@app.get("/", response_model=ServiceInfo, tags=["meta"])
def root() -> ServiceInfo:
    return ServiceInfo()


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    config = _state.config
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="moderator is not initialised yet",
        )
    return HealthResponse(
        status="ok",
        model_type=config.model_type,
        model_name=config.model_name,
        base_url=config.base_url,
        auth_required=bool(os.getenv("API_AUTH_TOKEN", "").strip()),
    )


@app.post(
    "/moderate",
    response_model=ModerateResponse,
    tags=["moderate"],
    dependencies=[Depends(require_auth)],
)
async def moderate(req: ModerateRequest) -> ModerateResponse:
    out = await _provider().acall(_state.system_prompt, req.text)
    return _to_response(out)


@app.post(
    "/moderate/batch",
    response_model=BatchModerateResponse,
    tags=["moderate"],
    dependencies=[Depends(require_auth)],
)
async def moderate_batch(req: BatchModerateRequest) -> BatchModerateResponse:
    provider = _provider()
    config = _state.config
    cap = max(1, config.concurrency) if config is not None else 8
    sem = asyncio.Semaphore(cap)

    async def _one(text: str) -> ModelOutput:
        async with sem:
            return await provider.acall(_state.system_prompt, text)

    outputs = await asyncio.gather(*[_one(t) for t in req.texts])
    results = [_to_response(o) for o in outputs]
    total_prompt = sum(o.prompt_tokens for o in outputs)
    total_completion = sum(o.completion_tokens for o in outputs)
    return BatchModerateResponse(
        results=results,
        usage=_usage_for(total_prompt, total_completion),
    )


# ---------- module entrypoint ----------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="moderator-api",
        description="Run the moderator as an HTTP service (uvicorn + FastAPI).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload (development only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "uvicorn is not installed. Run `pip install -r requirements.txt`."
        ) from e
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
