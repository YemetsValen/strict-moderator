# Moderator

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
│   ├── main.py                  # CLI entry point
│   ├── evaluator.py             # dataset → predictions → metrics
│   ├── metrics.py               # accuracy/precision/recall/F1 from scratch
│   ├── model.py                 # OpenAI + mock providers (Strategy pattern)
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
