# Moderator

> 🇷🇺 **Не разработчик?** Пошаговая инструкция для запуска "из коробки" —
> [INSTRUCTIONS_RU.md](INSTRUCTIONS_RU.md). Подходит для запуска на любой ОС
> без опыта работы с терминалом.

A flexible evaluation framework for **LLM-based content moderation**.
Plug in a JSONL dataset, pick a model provider, and run accuracy /
precision / recall / F1 — with optional **macro / micro / weighted**
averaging.

The bundled example evaluates a moderator for a **dating-app**:
ten short messages across four behaviours (friendly, spam,
off-platform redirect, hidden aggression) with a strict-JSON system
prompt that is hardened against simple jailbreak attacks.

```
.
├── data/
│   └── sample_dataset.jsonl     # 10 example messages, 4 categories
├── prompts/
│   └── system_prompt.txt        # the moderator's system prompt
├── src/
│   ├── main.py                  # CLI entry point (batch evaluation)
│   ├── api.py                   # FastAPI HTTP service
│   ├── evaluator.py             # dataset → predictions → metrics
│   ├── metrics.py               # accuracy/precision/recall/F1 from scratch
│   ├── model.py                 # OpenAI + Anthropic + mock providers
│   └── utils.py                 # config, JSONL I/O, robust JSON parsing
├── tests/                       # pytest unit + jailbreak tests
├── config.yaml                  # everything you can tune lives here
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/YemetsValen/Moderator.git
cd Moderator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --config config.yaml
```

By default `config.yaml` uses the **mock provider** — a deterministic
rule-based classifier that needs no API key. Output is written to
`reports/last_run.json` and a summary is printed to stdout.

### Picking a provider

Switch providers entirely from `config.yaml` — no code changes:

**OpenAI**
```yaml
model_type: openai
model_name: gpt-4o-mini
```
```bash
export OPENAI_API_KEY=sk-...
python -m src.main
```

**Anthropic Claude**
```yaml
model_type: anthropic
model_name: claude-3-5-haiku-latest    # or claude-3-5-sonnet-latest, etc.
```
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.main
```

**DeepSeek / OpenRouter / Together / Ollama** (any OpenAI-compatible API)
```yaml
model_type: openai
model_name: deepseek-chat
base_url: https://api.deepseek.com/v1
```
```bash
export OPENAI_API_KEY=<deepseek-key>
python -m src.main
```

You can also override any of `dataset_path`, `model_type`, or `output_path`
on the command line:

```bash
python -m src.main --dataset data/my_data.jsonl --model-type openai --output reports/openai.json
```

## HTTP service (FastAPI)

The same provider stack is also exposed as an HTTP API for use from other
services / front-ends:

```bash
pip install -r requirements.txt
# pick a provider in config.yaml + export the matching API key, then:
uvicorn src.api:app --host 0.0.0.0 --port 8000
# or:
python -m src.api --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the auto-generated Swagger UI.

| Method & path           | Body                                  | Returns                                      |
|-------------------------|---------------------------------------|----------------------------------------------|
| `GET /`                 | —                                     | Service info (version, links to /docs)       |
| `GET /health`           | —                                     | `{status, model_type, model_name, ...}`      |
| `POST /moderate`        | `{"text": "..."}`                     | `{verdict, category, confidence, reason, …}` |
| `POST /moderate/batch`  | `{"texts": ["...", "..."]}` (≤ 64)    | `{"results": [...]}`                         |

Example:

```bash
curl -s -X POST http://localhost:8000/moderate \
     -H 'Content-Type: application/json' \
     -d '{"text":"hey, message me on whatsapp +380501234567"}'
# → {"verdict":"BLOCK","category":"gray_platform_switch","confidence":0.85,...}
```

**Optional bearer auth.** Set `API_AUTH_TOKEN=<secret>` in the environment
and every `/moderate*` request must carry `Authorization: Bearer <secret>`.
`/health` is always open so liveness/readiness probes don't need a token.
Leave `API_AUTH_TOKEN` unset for local dev — the API will accept all
requests.

The service loads `config.yaml`, the system prompt, and the chosen
provider **once at startup** via FastAPI's `lifespan` hook, so each
request is just one provider call plus JSON parsing. Override the config
path with `MODERATOR_CONFIG=/path/to/config.yaml`.

## Example output

```
============================================================
  Examples            : 10
  Category accuracy   : 9/10
  Verdict accuracy    : 1.000
  Block recall        : 1.000
  Parse failures      : 0
  Duration (sec)      : 0.004
------------------------------------------------------------
  accuracy            : 0.900
  precision:
      ok                       : 1.000
      spam                     : 1.000
      gray_platform_switch     : 0.750
      hidden_aggression        : 1.000
      other                    : 0.000
      [macro]                  : 0.750
      [micro]                  : 0.900
  ...
============================================================
```

## Customising

### Dataset

JSONL, one example per line:

```json
{"text": "your message", "label": "ok"}
```

Point `dataset_path` at it (or pass `--dataset`). Labels must be a subset
of `labels` in `config.yaml`.

### System prompt

Plain text at `prompts/system_prompt.txt`. The path is configurable via
`prompt_path` in `config.yaml`. The prompt should instruct the model to
return strict JSON of shape:

```json
{"verdict": "ALLOW|BLOCK", "reason": "...", "category": "...", "confidence": 0.0}
```

Robust parsing in `src/utils.py:extract_first_json_object` recovers from
common LLM mistakes (markdown fences, extra prose); a parse failure
falls back to `BLOCK` with `confidence=0.0` and is recorded in
`summary.parse_failures`.

### Categories

Add a new label by listing it under `labels:` in `config.yaml`. Decide
whether it should map to BLOCK by also adding it to `block_labels:`.
**No code changes required** — the metric and verdict logic both read
the lists from config.

### Metrics

The metric registry lives in `src/metrics.py`:

```python
METRIC_REGISTRY: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}
```

To add a new metric:

1. Implement `def my_metric(y_true, y_pred, labels) -> dict[str, float] | float`
2. Register it: `METRIC_REGISTRY["my_metric"] = my_metric`
3. List `my_metric` under `metrics:` in `config.yaml`

### Model provider

`src/model.py` defines a `ModelProvider` abstract base. Three
implementations ship:

- `OpenAIProvider` — Chat Completions with `response_format={"type": "json_object"}`. Honours `base_url` so the same class drives DeepSeek, OpenRouter, Together, Ollama, etc.
- `AnthropicProvider` — Claude Messages API. The system prompt is passed at the top level (per Anthropic's API) and text-blocks of the response are concatenated before parsing.
- `MockProvider` — deterministic regex rules; useful for CI and as a baseline.

To add a brand-new provider (Cohere, Mistral, Google Gemini, …):
subclass `ModelProvider`, implement `call(system_prompt, text) -> ModelOutput`, then extend `build_provider()` with the new key.

## Jailbreak resistance

The system prompt explicitly:

- Treats the user message as **data**, not instructions
- Forces a single-line JSON output, no markdown / fences
- Refuses to "ignore previous instructions" attacks
- Flags charitable / urgent / romantic framing as **non-overriding** —
  if the substance is spam or off-platform redirect, it stays BLOCKed

Two regression cases live in `tests/test_evaluator.py`:

1. A polite WhatsApp redirect
2. A fake "donation for a children's hospital" with a `bit.ly` link

Both must be BLOCKed by the rule-based mock; if you swap in OpenAI or
another LLM, run `pytest` to confirm the same behaviour holds.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -v
```

CI runs all three on every push and PR (`.github/workflows/ci.yml`),
plus a `smoke` job that executes the full evaluation pipeline against
the bundled dataset using the mock provider.

## License

MIT — see [LICENSE](LICENSE).
