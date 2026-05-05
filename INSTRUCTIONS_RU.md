# Strict Moderator — пошаговая инструкция для запуска

Эта инструкция написана так, чтобы её мог пройти человек, который раньше
ничего не запускал из терминала. Если получится Hello World — получится и
этот проект.

Время: **~10 минут** при первом запуске, потом ~10 секунд на прогон.

---

## Что вы получите в итоге

Скрипт прочитает 10 примеров сообщений из `data/sample_dataset.jsonl`
(или из вашего датасета — об этом ниже) и для каждого выведет вердикт
**ALLOW** (пропустить) или **BLOCK** (заблокировать). В конце покажет
сводку: точность, полноту по каждой категории, F1.

Пример того, что должно появиться в окне:

```
[  1/10] ✓ expected=ok                     got=ok                     verdict=ALLOW conf=0.70
...
============================================================
  Examples            : 10
  Category accuracy   : 10/10
  Verdict accuracy    : 1.000
  Block recall        : 1.000
  Parse failures      : 0
============================================================
```

Если вы видите что-то похожее — всё работает.

---

## Шаг 1. Установить Python 3.10 или новее

Скрипт написан для **Python 3.10+**. Минимум — 3.10, рекомендуется 3.12.

### Windows
1. Открыть https://www.python.org/downloads/windows/
2. Скачать **Windows installer (64-bit)** для 3.12.x
3. Запустить установщик. **Поставить галочку "Add python.exe to PATH"** на первом экране.
4. Нажать "Install Now".
5. Открыть **PowerShell** (Win + R → `powershell` → Enter).
6. Проверить: `python --version`. Должно вывести `Python 3.12.x`.

> Если пишет "python is not recognized" — Python поставился без галочки PATH.
> Удалите его (Параметры → Приложения → Python → Удалить) и поставьте заново
> с включённой галочкой "Add to PATH".

### macOS
1. Установить [Homebrew](https://brew.sh) если его нет (одна команда с сайта).
2. В Терминале: `brew install python@3.12`
3. Проверить: `python3 --version` → `Python 3.12.x`.

### Linux (Ubuntu / Debian)
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip
python3 --version
```

---

## Шаг 2. Скачать проект

В PowerShell / Терминале выполнить:

```bash
git clone https://github.com/YemetsValen/strict-moderator.git
cd strict-moderator
```

Если у вас нет `git`:
- **Windows**: установите https://git-scm.com/download/win (опции по умолчанию).
- **macOS**: `brew install git`
- **Linux**: `sudo apt install git`

> **Альтернатива без git:** на странице репозитория есть зелёная кнопка
> **Code → Download ZIP**. Скачайте, распакуйте, в PowerShell зайдите в
> папку: `cd Downloads\strict-moderator-main` (имя может отличаться).

---

## Шаг 3. Создать виртуальное окружение

Это нужно, чтобы установленные пакеты не сломали другие Python-проекты на
вашем компьютере. Виртуальное окружение — это просто отдельная папка
`.venv/` со своими пакетами.

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Если PowerShell ругается **"running scripts is disabled on this system"** —
> один раз выполните: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> и подтвердите. Потом снова `Activate.ps1`.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

После активации в начале строки терминала появится `(.venv)` — это значит,
что окружение работает.

---

## Шаг 4. Установить зависимости

```bash
pip install -r requirements.txt
```

Это займёт 30–60 секунд. В конце должно быть `Successfully installed ...`.

> Если пишет **"pip is not recognized"** — выполните `python -m pip install -r requirements.txt`
> вместо `pip install ...`.

---

## Шаг 5. Запустить пайплайн (без API-ключей)

```bash
python -m src.main --config config.yaml
```

В `config.yaml` по умолчанию стоит `model_type: mock` — это
**не настоящая нейросеть**, а встроенный детерминированный классификатор
по правилам. Он нужен для проверки, что у вас всё установлено
правильно — никаких ключей и интернета не требует.

Если на этом шаге всё проходит — установка прошла успешно. Идём дальше.

---

## Шаг 6. Подключить настоящую модель (по желанию)

Скрипт умеет работать с тремя провайдерами. Вам нужен ровно один — тот,
от которого у вас есть API-ключ.

### Вариант A: OpenAI (GPT-4o, GPT-4.1, o3-mini, …)

1. Получить ключ: https://platform.openai.com/api-keys → Create new secret key.
2. Открыть `config.yaml` в Блокноте, поменять:
   ```yaml
   model_type: openai
   model_name: gpt-4o-mini    # или gpt-4o, gpt-4.1
   ```
3. В терминале (где вы видите `(.venv)`):

   **Windows PowerShell:**
   ```powershell
   $env:OPENAI_API_KEY = "sk-..."
   python -m src.main
   ```

   **macOS / Linux:**
   ```bash
   export OPENAI_API_KEY="sk-..."
   python -m src.main
   ```

### Вариант B: Anthropic (Claude Haiku / Sonnet / Opus)

1. Получить ключ: https://console.anthropic.com/settings/keys
2. В `config.yaml`:
   ```yaml
   model_type: anthropic
   model_name: claude-3-5-haiku-latest    # или claude-3-5-sonnet-latest
   ```
3. В терминале:

   **Windows PowerShell:**
   ```powershell
   $env:ANTHROPIC_API_KEY = "sk-ant-..."
   python -m src.main
   ```

   **macOS / Linux:**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   python -m src.main
   ```

### Вариант C: DeepSeek (или любой OpenAI-совместимый API)

DeepSeek в 5–10 раз дешевле OpenAI, для модерации этого хватает с запасом.

1. Получить ключ: https://platform.deepseek.com/api_keys
2. В `config.yaml`:
   ```yaml
   model_type: openai          # да, openai — DeepSeek совместим с OpenAI API
   model_name: deepseek-chat
   base_url: https://api.deepseek.com/v1
   ```
3. В терминале:
   ```bash
   export OPENAI_API_KEY="<ваш-ключ-deepseek>"
   python -m src.main
   ```

---

## Шаг 7. Подменить датасет на свой

В файле `data/sample_dataset.jsonl` лежит 10 примеров. Это формат **JSONL** —
по одной JSON-строке на пример:

```json
{"text": "Привет, как дела?", "label": "ok"}
{"text": "Курс по крипте за $5000", "label": "spam"}
```

`label` обязан быть одним из значений в `config.yaml → labels:`. По умолчанию:
`ok`, `spam`, `gray_platform_switch`, `hidden_aggression`, `other`.

Чтобы добавить свою категорию (например, `sexual`):
1. Допишите её в `config.yaml → labels:`
2. Если она должна блокироваться — добавьте в `config.yaml → block_labels:`
3. Никаких изменений в коде делать не нужно.

Затем запустить с новым датасетом:
```bash
python -m src.main --dataset path/to/your.jsonl
```

---

## Если не работает — таблица типовых ошибок

| Что вы видите | Что это значит | Что делать |
|---|---|---|
| `python: command not found` | Python не установлен или не в PATH | Шаг 1 заново; на Windows — переустановить с галочкой "Add to PATH" |
| `pip: command not found` | pip не виден | Использовать `python -m pip install ...` вместо `pip install ...` |
| `ModuleNotFoundError: No module named 'yaml'` | Зависимости не поставились | Активировать `.venv` (Шаг 3) и заново `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'src'` | Запуск из неправильной папки | `cd` в папку `strict-moderator/` (там где лежит `config.yaml`) |
| `FileNotFoundError: config.yaml` | Файл не найден | Вы запускаете не из той папки. Сделать `pwd` (Mac/Linux) или `Get-Location` (Windows), убедиться что вы в папке `strict-moderator` |
| `OPENAI_API_KEY is not set` | Не выставили переменную окружения | Шаг 6, не забудьте кавычки вокруг ключа |
| `unknown model_type: 'antrophic'` | Опечатка в `config.yaml` | Должно быть ровно `anthropic`, `openai` или `mock` |
| `running scripts is disabled` (Windows) | PowerShell блокирует скрипты | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` один раз |
| `git: command not found` | Git не установлен | Поставить https://git-scm.com или скачать ZIP вместо клонирования |
| Падает на загрузке `requirements.txt`, ругается на `anthropic` или `openai` | У вас Python < 3.10 или нет интернета | Проверить `python --version`, проверить интернет |
| Запустилось, но `Block recall: 0.5` | Это **не ошибка** — модель ошиблась на половине BLOCK-кейсов | Поменять модель на более мощную (Sonnet → Opus, gpt-4o-mini → gpt-4o) или подкрутить `prompts/system_prompt.txt` |

---

## Если ничего не помогло

Пришлите автору:

1. Что именно вы запустили (полную команду).
2. **Полный текст ошибки** (всё что напечаталось в красном/жёлтом — целиком, не "написало что-то не работает", а **скриншот** или скопированный текст).
3. Какая ОС: Windows / macOS / Linux.
4. Версия Python: `python --version`.
5. Содержимое `config.yaml` (ничего секретного там нет, ключи лежат в переменных окружения).

С этими пятью пунктами проблему обычно чинят за 2 минуты.

---

## Что дальше

- Полный README с архитектурой и API: [README.md](README.md)
- Тесты, как добавлять метрики, как добавлять провайдеров — там же.
