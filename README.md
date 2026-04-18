# TwoTower Project

Небольшой учебный проект рекомендательной системы на PyTorch с архитектурой `two-tower`. Репозиторий собирает данные из CSV, готовит бинарный таргет по кликам, обучает две embedding-башни, считает простые метрики и сохраняет чекпойнт модели.

README ниже описывает проект в его текущем состоянии, по реальному коду в репозитории.

## Что делает проект

- Загружает пользователей, баннеры и историю взаимодействий из `data/raw/*.csv`.
- Строит отображения `user_id -> index` и `banner_id -> index`.
- Формирует бинарную метку `label = 1`, если `clicks > 0`, иначе `0`.
- Ограничивает обучающую выборку до `250_000` взаимодействий с балансировкой классов.
- Делит данные по времени на `train/valid/test`.
- Обучает `TwoTower`-модель на PyTorch.
- Считает `loss`, `accuracy` и `recall@k`.
- Показывает пример рекомендаций для нескольких пользователей.
- Сохраняет модель в `artifacts/twotower_model.pth`.

## Структура репозитория

```text
.
├── main.py
├── pyproject.toml
├── README.md
├── artifacts/
│   └── twotower_model.pth
├── data/
│   └── raw/
│       ├── users.csv
│       ├── banners.csv
│       └── interactions.csv
├── src/
│   ├── data.py
│   └── сonfig.py
└── twotower/
    ├── __init__.py
    ├── config.py
    ├── core.py
    ├── item_tower.py
    └── user_tower.py
```

## Быстрый контекст по файлам

- `main.py` — точка входа: загрузка данных, подготовка, обучение, оценка, предсказания и сохранение модели.
- `src/data.py` — чтение CSV, подготовка interactions, семплирование и временной сплит.
- `src/сonfig.py` — конфиг путей к данным и артефактам.
- `twotower/config.py` — гиперпараметры модели и обучения.
- `twotower/user_tower.py` — user tower.
- `twotower/item_tower.py` — item tower.
- `twotower/core.py` — основной класс `TwoTower`: `fit`, `predict`, `evaluate`, `save_model`, `load_model`.

## Данные

В репозитории есть локальный датасет:

- `data/raw/users.csv` — `5000 x 14`
- `data/raw/banners.csv` — `3000 x 14`
- `data/raw/interactions.csv` — `1_499_110 x 6`

Период interactions:

- от `2026-01-01`
- до `2026-03-31`

### Колонки `users.csv`

`user_id`, `age`, `gender`, `city_tier`, `device_os`, `platform`, `income_band`, `activity_segment`, `interest_1`, `interest_2`, `interest_3`, `country`, `signup_days_ago`, `is_premium`

### Колонки `banners.csv`

`banner_id`, `brand`, `category`, `subcategory`, `banner_format`, `campaign_goal`, `target_gender`, `target_age_min`, `target_age_max`, `cpm_bid`, `quality_score`, `created_at`, `is_active`, `landing_page`

### Колонки `interactions.csv`

`event_date`, `user_id`, `banner_id`, `impressions`, `clicks`, `ctr`

### Как формируется таргет

Из interactions используются поля `event_date`, `user_id`, `banner_id`, `clicks`.

Правило разметки:

```text
clicks > 0  -> label = 1
clicks == 0 -> label = 0
```

После подготовки и ограничения `max_samples=250_000` текущий пайплайн получает:

- `250_000` строк после семплирования
- `125_000` позитивных примеров
- `125_000` негативных примеров
- `175_000` строк в `train`
- `50_000` строк в `valid`
- `25_000` строк в `test`

## Архитектура модели

Проект реализует классическую схему `two-tower`, но в очень минимальном виде.

### User tower

```text
Embedding(user_id) -> Linear -> ReLU -> LayerNorm
```

### Item tower

```text
Embedding(banner_id) -> Linear -> ReLU -> LayerNorm
```

### Scoring

Эмбеддинги нормализуются, после чего score считается как скалярное произведение:

```text
dot(normalize(user_embedding), normalize(item_embedding))
```

### Loss

Для обучения используется:

- `Adam`
- `BCEWithLogitsLoss`

## Важное ограничение текущей реализации

Хотя в `users.csv` и `banners.csv` много признаков, модель их сейчас не использует. Фактически в обучение попадают только:

- `user_id`
- `banner_id`
- бинарная метка из `clicks`

Это значит:

- проект сейчас ближе к ID-based baseline, чем к полноценной feature-rich recommender system;
- таблицы пользователей и баннеров нужны главным образом для построения индексов;
- cold-start для новых `user_id` и `banner_id` не поддержан;
- неизвестные пользователи и баннеры при подготовке/evaluate/predict отфильтровываются.

## Поток выполнения

Текущий сценарий `main.py`:

1. Создаёт `Config()` из `src/сonfig.py`.
2. Загружает CSV по путям из конфига.
3. Строит индексы пользователей и баннеров.
4. Подготавливает interactions и считает `label`.
5. Делит данные на `train/valid/test`.
6. Создаёт `TwoTower()` с дефолтным `TwoTowerConfig`.
7. Вызывает `fit(train_df, valid_df)`.
8. Вызывает `evaluate(test_df)`.
9. Строит `top-k` рекомендации для первых трёх пользователей из `users.csv`.
10. Сохраняет чекпойнт в `artifacts/twotower_model.pth`.

## Конфигурация

### Конфиг путей и данных

Файл: `src/сonfig.py`

```python
Config(
    users_path="data/raw/users.csv",
    items_path="data/raw/banners.csv",
    interactions_path="data/raw/interactions.csv",
    model_save_path="artifacts/twotower_model.pth",
    max_samples=250_000,
    seed=42,
)
```

### Конфиг модели

Файл: `twotower/config.py`

```python
TwoTowerConfig(
    user_embedding_dim=64,
    item_embedding_dim=64,
    hidden_dim=64,
    learning_rate=1e-3,
    batch_size=2048,
    epochs=10,
    validation_ratio=0.2,
    test_ratio=0.1,
    max_samples=250_000,
    max_eval_users=500,
    top_k=10,
    seed=42,
    device="cpu",
)
```

## Метрики

Во время обучения и оценки проект считает:

- `train_loss`
- `valid_loss`
- `valid_accuracy`
- `test_loss`
- `test_accuracy`
- `recall_at_k`

`recall_at_k` считается только по пользователям с позитивными событиями в evaluation-части и ограничивается первыми `max_eval_users`.

## Установка

Требования:

- Python `>=3.11`
- `torch`
- `pandas`
- `rich`

Зависимости описаны в `pyproject.toml`.

Через `uv`:

```bash
uv sync
```

Через `venv` и `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Запуск

```bash
python main.py
```

или

```bash
uv run python main.py
```

На выходе скрипт:

- печатает прогресс загрузки данных;
- логирует метрики по эпохам;
- печатает итоговые метрики на тесте;
- печатает пример рекомендаций;
- сохраняет модель в `artifacts/twotower_model.pth`.

## Работа с чекпойнтом

Класс `TwoTower` умеет:

- сохранять модель через `save_model(path)`
- загружать модель через `load_model(path)`

В чекпойнт кладутся:

- конфиг модели;
- `state_dict`;
- словари индексации пользователей и баннеров;
- обратные списки `idx -> id`;
- история обучения `train_history`

## Неочевидные особенности проекта

- В `src/сonfig.py` первая буква в имени файла — кириллическая `с`, а не латинская `c`. Импорты в проекте уже написаны под это имя: `from src.сonfig import Config`.
- Функция `split_interactions()` умеет принимать `validation_ratio` и `test_ratio`, но в `main.py` она вызывается без явной передачи значений, поэтому используются её дефолты `0.2` и `0.1`.
- `TwoTower()` в `main.py` создаётся без передачи внешнего конфига, поэтому обучение идёт на дефолтных значениях из `TwoTowerConfig`.
- Отдельных тестов, CLI-интерфейса и конфигурации через аргументы командной строки в репозитории сейчас нет.

## Для чего этот проект подходит

- как минимальный baseline для задачи рекомендаций;
- как учебный пример `two-tower` на PyTorch;
- как стартовая точка перед добавлением реальных user/item features, negative sampling, retrieval-индекса и offline/online метрик.

## Что логично развивать дальше

- подключить табличные признаки пользователей и баннеров в обе башни;
- вынести параметры запуска в единый конфиг;
- добавить CLI или `.env`/YAML-конфиг;
- покрыть пайплайн тестами;
- разделить retrieval и ranking;
- добавить более информативные метрики, например `precision@k`, `map@k`, `ndcg@k`.
