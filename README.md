# TwoTower Project

Минимальный baseline рекомендательной системы на базе архитектуры `two-tower`, реализованный на PyTorch.

Проект обучает две отдельные башни:
- `user tower` строит embedding пользователя
- `item tower` строит embedding баннера

Сходство между пользователем и баннером считается через скалярное произведение нормализованных embedding-векторов. На выходе модель умеет:
- обучаться на истории взаимодействий
- оценивать качество на тесте
- выдавать `top-k` рекомендаций для пользователей
- сохранять и загружать обученный чекпойнт

## Что лежит в проекте

```text
.
├── main.py                  # точка входа: обучение, оценка, предсказания, сохранение модели
├── pyproject.toml           # зависимости проекта
├── src/
│   └── data.py              # загрузка и подготовка данных
├── twotower/
│   ├── __init__.py
│   ├── config.py            # dataclass-конфиг модели
│   ├── core.py              # обучение, инференс, метрики, сериализация
│   ├── user_tower.py        # user tower
│   └── item_tower.py        # item tower
├── data/
│   └── raw/
│       ├── users.csv
│       ├── banners.csv
│       └── interactions.csv
└── artifacts/
    └── twotower_model.pth   # сохраненный чекпойнт модели
```

## Требования

- Python `3.11+`
- PyTorch
- pandas
- rich

Зависимости описаны в `pyproject.toml`.

## Установка

Если используешь `uv`:

```bash
uv sync
```

Если используешь стандартный `venv` и `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Запуск

Основной сценарий запуска:

```bash
python main.py
```

Если проект установлен через `uv`:

```bash
uv run python main.py
```

Во время запуска скрипт:
1. загружает данные из `data/raw/*.csv`
2. готовит interactions и делит их по времени на `train/valid/test`
3. создает модель `TwoTower`
4. обучает модель на `train_df` и `valid_df`
5. считает метрики на `test_df`
6. печатает пример рекомендаций для нескольких пользователей
7. сохраняет веса в `artifacts/twotower_model.pth`

## Формат данных

### `users.csv`

Таблица пользователей. В текущем датасете встречаются поля вроде:

- `user_id`
- `age`
- `gender`
- `city_tier`
- `device_os`
- `platform`
- `income_band`
- `activity_segment`
- `interest_1`, `interest_2`, `interest_3`
- `country`
- `signup_days_ago`
- `is_premium`

### `banners.csv`

Таблица объектов рекомендаций. В текущем датасете встречаются поля:

- `banner_id`
- `brand`
- `category`
- `subcategory`
- `banner_format`
- `campaign_goal`
- `target_gender`
- `target_age_min`
- `target_age_max`
- `cpm_bid`
- `quality_score`
- `created_at`
- `is_active`
- `landing_page`

### `interactions.csv`

История взаимодействий пользователей с баннерами. Для обучения критичны поля:

- `event_date`
- `user_id`
- `banner_id`
- `clicks`

Дополнительно в датасете могут быть:

- `impressions`
- `ctr`

Таргет формируется так:

```text
label = 1, если clicks > 0
label = 0, если clicks == 0
```

## Как устроена модель

### User tower

Для пользователя используется последовательность:

```text
Embedding(user_id) -> Linear -> ReLU -> LayerNorm
```

### Item tower

Для баннера используется такая же схема:

```text
Embedding(banner_id) -> Linear -> ReLU -> LayerNorm
```

### Scoring

После этого обе башни возвращают embedding одинаковой размерности, и score пары считается как:

```text
dot(normalize(user_embedding), normalize(item_embedding))
```

## Текущий пайплайн обучения

1. Из `users.csv` и `banners.csv` строятся отображения `id -> index`.
2. `interactions.csv` приводится к нужным типам.
3. Взаимодействия сортируются по `event_date`.
4. Если данных больше `max_samples`, выполняется подвыборка с балансировкой позитивных и негативных примеров.
5. Данные делятся по времени на `train`, `valid`, `test`.
6. `fit()` принимает готовые `train_df` и `valid_df`.
7. `evaluate()` принимает отдельный `test_df`.
8. Модель обучается с `Adam` и `BCEWithLogitsLoss`.

## Конфигурация

Основные параметры находятся в `twotower/config.py`.

Параметры по умолчанию:

```python
TwoTowerConfig(
    user_embedding_dim=64,
    item_embedding_dim=64,
    hidden_dim=64,
    learning_rate=1e-3,
    batch_size=2048,
    epochs=3,
    validation_ratio=0.2,
    test_ratio=0.1,
    max_samples=250_000,
    max_eval_users=500,
    top_k=10,
    seed=42,
    device=None,
)
```

В `main.py` часть параметров переопределяется, например:
- `epochs=10`
- `top_k=5`

## Метрики

Проект считает:

- `valid_loss`
- `valid_accuracy`
- `test_loss`
- `test_accuracy`
- `recall_at_k`

Во время обучения логируются `valid_loss` и `valid_accuracy`.

В `evaluate(test_df)` считаются `test_loss`, `test_accuracy` и `recall_at_k`.

`recall_at_k` считается по пользователям из тестового набора, у которых есть положительные взаимодействия.

## Использование из кода

Пример программного использования:

```python
from src.data import fit_id_mappings, load_data, prepare_interactions, split_interactions
from twotower import TwoTower, TwoTowerConfig

data_config = {
    "users_path": "data/raw/users.csv",
    "items_path": "data/raw/banners.csv",
    "interactions_path": "data/raw/interactions.csv",
    "max_samples": 250_000,
    "seed": 42,
}

users_df, items_df, interactions_df = load_data(data_config)
user_idx, item_idx = fit_id_mappings(users_df, items_df)
interactions = prepare_interactions(interactions_df, user_idx, item_idx, data_config)
train_df, valid_df, test_df = split_interactions(interactions)

model = TwoTower(TwoTowerConfig())
model.fit(train_df, valid_df)

metrics = model.evaluate(test_df, top_k=5)
predictions = model.predict(user_ids=[1, 2, 3], top_k=5)

model.save_model("artifacts/twotower_model.pth")
```

Загрузка сохраненной модели:

```python
from twotower import TwoTower

model = TwoTower().load_model("artifacts/twotower_model.pth")
predictions = model.predict(user_ids=[1, 2, 3], top_k=5)
```

## Ограничения текущего baseline

Это именно baseline-реализация, и у нее есть важные ограничения:

- модель ожидает, что train/valid/test уже подготовлены на уровне пайплайна данных
- используются только `user_id` и `banner_id`
- признаки из `users.csv` и `banners.csv` пока не участвуют в обучении
- нет отдельного production-ready пайплайна инференса
- нет тестов
- нет логирования экспериментов
- нет конфигурации через CLI или env
- нет hard negative sampling и более продвинутых retrieval-loss функций

Из-за этого качество может быть ограничено, особенно на холодных пользователях и новых баннерах.

## Идеи для развития

- добавить user/item features поверх ID embeddings
- ввести категориальные и числовые признаки
- добавить negative sampling для retrieval-задачи
- использовать более подходящие ranking-метрики
- вынести обучение и инференс в отдельные команды
- добавить тесты и описание экспериментов
- сохранить preprocessing и конфиг вместе с моделью более формально

## Полезные файлы

- `main.py` — быстрый end-to-end запуск
- `src/data.py` — загрузка и подготовка данных
- `twotower/core.py` — основная логика модели
- `twotower/config.py` — параметры модели

## Статус

Проект находится на стадии рабочего прототипа / baseline для дальнейших экспериментов с retrieval-рекомендациями.
