# TwoTower Project

`TwoTower Project` — это репозиторий с библиотечной реализацией two-tower модели для retrieval-рекомендаций.
Текущий целевой public API модели:

- `fit(...)`
- `predict(...)`
- `evaluate(...)`
- `save_model(...)`
- `load_model(...)`

README в этом проекте считается живой документацией. Если мы меняем public API, сценарии запуска, формат данных, чекпоинт или структуру модулей, README нужно обновлять в том же изменении.

## Текущее состояние

Сейчас проект уже разделен на библиотечное ядро и прикладную обвязку:

- `twotower/` — библиотека и public API модели.
- `src/` — прикладная обвязка для загрузки данных и API-сервера.
- `main.py` — локальный training/evaluation сценарий.
- `tests/` — unit-тесты на `fit`, `predict`, `evaluate`, `save_model`, `load_model`.

Архитектурная идея текущей версии: сервисные операции вынесены из `core.py` в отдельные модули (`fit.py`, `predict.py`, `evaluate.py`, `save_model.py`, `load_model.py`) и общаются с моделью через минимальные Protocol-интерфейсы. `TwoTower` напрямую реализует эти протоколы — прослоек-адаптеров нет. Это позволяет тестировать `fit`, `predict`, `evaluate`, `save/load` без поднятия полной модели.

Packaging-контракт текущей версии:

- build backend: `setuptools`;
- importable package в wheel ограничен `twotower`;
- прикладная обвязка из `src/` устанавливается через `pip install twotower[app]` и не входит в базовый пакет;
- dev-зависимости (`pytest`, `ruff`, `mypy`) устанавливаются через `pip install twotower[dev]`.

## Структура библиотеки

Основные файлы в `twotower/`:

- `core.py` — фасад `TwoTower`, оркестрация; не содержит логику подготовки данных.
- `config.py` — `TwoTowerConfig`.
- `preprocessing.py` — чистые функции подготовки данных: валидация, построение маппингов, фильтрация, сэмплинг.
- `fit.py` — обучение и pairwise negative sampling.
- `predict.py` — генерация top-k рекомендаций.
- `evaluate.py` — расчет loss и retrieval-метрик.
- `save_model.py` — сохранение чекпоинта.
- `load_model.py` — восстановление модели из чекпоинта.
- `data.py` — нормализация и temporal split interaction-данных.
- `features.py` — кодирование side-features: `FeatureConfig`, `MultiFeatureSpec`, `build_feature_tables`. Не содержит хардкода колонок; схема передаётся через `FeatureConfig`.
- `modules/` — нейросетевые tower-компоненты: `TwoTowerBase`, `UserTower`, `ItemTower`. Оба тауэра поддерживают scalar и multi-valued side features.

Публичный импорт:

```python
from twotower import TwoTower, TwoTowerConfig, FeatureConfig, MultiFeatureSpec, EarlyStopping
```

Package-level public exports: `TwoTower`, `TwoTowerConfig`, `FeatureConfig`, `MultiFeatureSpec`, `EarlyStopping`.
`TwoTowerBase` остается внутренней реализационной деталью и не считается частью стабильного внешнего API.

## Public API

### `TwoTowerConfig`

Основные параметры:

- `user_embedding_dim`
- `item_embedding_dim`
- `side_feature_embedding_dim`
- `hidden_dim`
- `retrieval_temperature`
- `symmetric_retrieval_loss`
- `observed_negative_sampling_ratio`
- `learning_rate`
- `batch_size`
- `epochs`
- `validation_ratio`
- `test_ratio`
- `max_samples`
- `eval_top_ks`
- `max_eval_users`
- `top_k`
- `eval_during_training` — если `True`, recall@k считается на валидации после каждой эпохи и пишется в историю (по умолчанию `True`)
- `seed`
- `device`

### `fit(...)`

Сигнатура:

```python
history = model.fit(
    X_train=train_df[["user_id", "banner_id"]],
    y_train=train_df["label"],
    X_valid=valid_df[["user_id", "banner_id"]],
    y_valid=valid_df["label"],
    users_df=users_df,                        # необязательно
    items_df=items_df,                        # необязательно
    user_feature_config=user_feature_config,  # необязательно
    item_feature_config=item_feature_config,  # необязательно
    early_stopping=EarlyStopping(patience=5, metric="valid_loss"),  # None — отключить
)
```

Ожидания:

- `X_train` и `X_valid` должны содержать `user_id` и `banner_id`.
- `y_train` и `y_valid` должны содержать бинарные метки.
- `users_df`, `items_df`, `user_feature_config`, `item_feature_config` необязательны, но должны передаваться все четыре вместе, если используются side-features.
- Колонки, указанные в `FeatureConfig`, должны быть уже подготовлены вызывающим кодом (библиотека выполняет только кодирование в словарь, числовые преобразования — на стороне приложения).

Результат:

- список словарей с историей обучения по эпохам: `epoch`, `train_loss`, `valid_loss`; при `eval_during_training=True` также `recall_at_<k>` для каждого значения из `eval_top_ks`.

### `predict(...)`

Сигнатура:

```python
predictions = model.predict(
    user_ids=[1, 2, 3],
    item_ids=None,
    top_k=10,
    exclude_seen=True,
    strict=False,
)
```

Поведение:

- если `user_ids=None`, модель предсказывает для первых известных пользователей, максимум для `10`;
- если `item_ids=None`, кандидатами становятся все известные items;
- если `exclude_seen=True`, уже виденные пользователем items исключаются;
- если `strict=False`, неизвестные `user_id` и `item_id` молча пропускаются;
- если `strict=True`, на неизвестные идентификаторы выбрасывается `ValueError`.

Формат результата:

```python
{
    1: [
        {"banner_id": 20, "score": 0.93},
        {"banner_id": 30, "score": 0.74},
    ]
}
```

На текущем этапе `predict_proba(...)` в public API нет.

### `evaluate(...)`

Сигнатура:

```python
metrics = model.evaluate(test_df, top_k=100)
```

Ожидания:

- `X_test` должен содержать `user_id` и `banner_id`;
- если в данных есть `label`, используется он;
- если в данных есть `clicks`, метка вычисляется из `clicks > 0`.

Типичные метрики:

- `test_loss`
- `recall_at_<k>`
- `popularity_recall_at_<k>`
- `recall_at_k`
- `popularity_recall_at_k`
- `test_input_rows`
- `test_rows_used`
- `test_rows_filtered`
- `test_unknown_user_rows`
- `test_unknown_item_rows`
- `test_positive_rate`
- `test_positive_pairs_used_for_loss`
- `test_eval_user_count`
- `test_rows_filtered_ratio`

### `save_model(...)` и `load_model(...)`

Сохранение:

```python
model.save_model("artifacts/twotower_model.pth")
```

Загрузка:

```python
loaded_model = TwoTower().load_model("artifacts/twotower_model.pth")
```

В чекпоинт сейчас сохраняются:

- `config`
- `state_dict`
- user/item id mappings
- `train_history`
- `seen_items_by_user`
- popularity ranking позитивных items
- metadata side-features

## Формат данных

### Raw interactions

Для функций из `twotower.data` ожидается таблица interaction-данных с колонками:

- `event_date`
- `user_id`
- `banner_id`
- `clicks`

`normalize_interactions(...)`:

- приводит `event_date` к `datetime`;
- приводит `user_id` и `banner_id` к `int`;
- создает `label = (clicks > 0).astype("float32")`.

### Train / valid / test

`split_interactions(...)` делает temporal split по `event_date` и требует минимум `3` уникальные даты.

После подготовки модель обычно работает с таблицами, где есть:

- `event_date`
- `user_id`
- `banner_id`
- `label`

### Side-features

Если в `fit(...)` передаются `users_df` и `items_df`, они используются для построения side-feature таблиц. Если side-features не нужны, оба аргумента можно не передавать.

## Быстрый пример использования

```python
import pandas as pd

from twotower import TwoTower, TwoTowerConfig, FeatureConfig, MultiFeatureSpec

users_df = pd.read_csv("data/raw/users.csv")
items_df = pd.read_csv("data/raw/banners.csv")
train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/valid.csv")
test_df = pd.read_csv("data/test.csv")

# Числовые преобразования — на стороне приложения, до передачи в модель
users_df["age_bucket"] = pd.cut(
    users_df["age"], bins=[-1, 24, 34, 44, 54, float("inf")],
    labels=["18_24", "25_34", "35_44", "45_54", "55_plus"],
).astype(str)

user_feature_config = FeatureConfig(
    scalar_features=("age_bucket", "gender", "city_tier"),
    multi_features=(
        MultiFeatureSpec("interest_ids", columns=("interest_1", "interest_2", "interest_3")),
    ),
)
item_feature_config = FeatureConfig(
    scalar_features=("brand", "category", "subcategory"),
)

model = TwoTower(
    TwoTowerConfig(
        epochs=3,
        batch_size=1024,
        top_k=50,
        seed=42,
        device="cpu",
    )
)

history = model.fit(
    X_train=train_df[["user_id", "banner_id"]],
    y_train=train_df["label"],
    X_valid=valid_df[["user_id", "banner_id"]],
    y_valid=valid_df["label"],
    users_df=users_df,
    items_df=items_df,
    user_feature_config=user_feature_config,
    item_feature_config=item_feature_config,
)

metrics = model.evaluate(test_df, top_k=50)
predictions = model.predict(user_ids=[101, 202], top_k=5, exclude_seen=True)

model.save_model("artifacts/twotower_model.pth")

restored_model = TwoTower().load_model("artifacts/twotower_model.pth")
restored_predictions = restored_model.predict(user_ids=[101], top_k=5)
```

## Локальный запуск

В репозитории используется локальное окружение `.venv`.

### Установка зависимостей

Если окружение уже собрано, команды можно запускать через:

```bash
./.venv/bin/python ...
```

### Запуск training-сценария

На текущем состоянии обвязка ожидает `PYTHONPATH=src`:

```bash
PYTHONPATH=src ./.venv/bin/python main.py
```

Обычный `./.venv/bin/python main.py` сейчас не подходит из-за текущего import layout в `src/`.

### Запуск API

FastAPI-приложение находится в `src/app/main.py`.

Пример запуска:

```bash
PYTHONPATH=src ./.venv/bin/python -m uvicorn src.app.main:app --reload
```

Эндпоинты:

- `GET /`
- `GET /health`
- `POST /recommendations`

Пример тела запроса:

```json
{
  "user_ids": [1, 2, 3],
  "item_ids": null,
  "top_k": 5,
  "exclude_seen": true,
  "strict": false
}
```

## Тесты

Запуск unit-тестов:

```bash
./.venv/bin/python -m pytest
```

Сейчас тестами покрыты:

- `fit`
- `predict`
- `evaluate`
- `save_model`
- `load_model`

## Сборка пакета

Собрать sdist и wheel можно так:

```bash
uv build --offline --no-build-isolation --out-dir /tmp/twotower-dist
```

Текущий `pyproject.toml` настроен так, чтобы собирать именно библиотечный пакет `twotower`, а не весь репозиторий целиком.
При этом `sdist` у `setuptools` может включать repo-level файлы вроде тестов, что нормально для исходного дистрибутива.

## Документационный контракт

README считается обязательной частью изменений. При любом изменении нужно проверить, не устарели ли следующие разделы:

- public API модели;
- сигнатуры и поведение методов;
- формат входных данных;
- структура чекпоинта;
- команды запуска;
- структура модулей;
- ограничения и known issues.

Правило для дальнейшей работы простое: изменили код, который меняет внешний контракт или способ работы проекта, сразу обновили README в том же изменении.
