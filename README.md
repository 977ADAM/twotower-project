# TwoTower Project

Учебный проект рекомендательной системы на PyTorch с архитектурой `two-tower`.
Сейчас это компактный baseline, который:

- читает CSV-данные пользователей, баннеров и взаимодействий;
- строит `id -> index` отображения;
- превращает `clicks` в бинарный таргет;
- балансирует и ограничивает выборку до `250_000` строк;
- делит данные по времени на `train/valid/test`;
- обучает модель из двух embedding-башен;
- считает `loss`, `accuracy` и `recall@k`;
- сохраняет чекпойнт модели в `artifacts/twotower_model.pth`.

README ниже описывает реальное текущее состояние репозитория по коду и локальным данным.

## Правило актуализации README

Этот файл считается главным источником контекста по проекту.

После любых заметных изменений в проекте нужно сразу обновлять `README.md`, если меняется хотя бы одно из следующего:

- структура репозитория;
- входные данные или формат датасета;
- конфиги, дефолтные гиперпараметры или пути;
- поведение pipeline обучения и инференса;
- архитектура модели;
- метрики, ограничения или способ запуска;
- текущее состояние проекта и ближайшие шаги.

Минимум, который нужно поддерживать в актуальном состоянии:

- разделы "Что делает проект", "Структура", "Pipeline", "Конфигурация", "Ограничения" и "Текущее состояние".

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
    ├── data.py
    ├── core.py
    ├── item_tower.py
    └── user_tower.py
```

## Что где лежит

- `main.py` - точка входа для полного сценария: загрузка данных, подготовка, обучение, оценка, пример рекомендаций, сохранение модели.
- `src/data.py` - data pipeline на уровне CSV: загрузка, построение ID-mapping, подготовка interactions, временной split.
- `src/сonfig.py` - конфиг путей к данным и общих параметров подготовки.
- `twotower/config.py` - гиперпараметры модели и обучения.
- `twotower/data.py` - общая реализация подготовки interactions и temporal split, которую используют и пакет, и `src/data.py`.
- `twotower/user_tower.py` - user tower.
- `twotower/item_tower.py` - item tower.
- `twotower/core.py` - основной класс `TwoTower` с `fit`, `predict`, `evaluate`, `save_model`, `load_model`.
- `artifacts/twotower_model.pth` - сохраненный чекпойнт.

## Текущий стек

- Python `>=3.11`
- `torch`
- `pandas`
- `rich`

Зависимости описаны в [pyproject.toml](/home/adam/projects/twotower-project/pyproject.toml:1).

## Данные

Локально в репозитории лежат:

- `data/raw/users.csv` - `5000 x 14`
- `data/raw/banners.csv` - `3000 x 14`
- `data/raw/interactions.csv` - `1_499_110 x 6`

Диапазон дат в `interactions.csv`:

- минимум: `2026-01-01`
- максимум: `2026-03-31`

Распределение по кликам в сырых interactions:

- строк с `clicks > 0`: `175_880`
- строк с `clicks == 0`: `1_323_230`

### Колонки `users.csv`

`user_id`, `age`, `gender`, `city_tier`, `device_os`, `platform`, `income_band`, `activity_segment`, `interest_1`, `interest_2`, `interest_3`, `country`, `signup_days_ago`, `is_premium`

### Колонки `banners.csv`

`banner_id`, `brand`, `category`, `subcategory`, `banner_format`, `campaign_goal`, `target_gender`, `target_age_min`, `target_age_max`, `cpm_bid`, `quality_score`, `created_at`, `is_active`, `landing_page`

### Колонки `interactions.csv`

`event_date`, `user_id`, `banner_id`, `impressions`, `clicks`, `ctr`

## Как формируется таргет

Из interactions используются поля:

- `event_date`
- `user_id`
- `banner_id`
- `clicks`

Правило разметки:

```text
clicks > 0  -> label = 1
clicks == 0 -> label = 0
```

## Pipeline данных

Текущий сценарий подготовки данных в [src/data.py](/home/adam/projects/twotower-project/src/data.py:1):

1. Загружает `users.csv`, `banners.csv`, `interactions.csv`.
2. Строит маппинги `user_id -> idx` и `banner_id -> idx`.
3. Оставляет только `event_date`, `user_id`, `banner_id`, `clicks`.
4. Приводит даты к `datetime`, идентификаторы к `int`.
5. Строит `label = (clicks > 0).astype("float32")`.
6. Отфильтровывает строки с неизвестными пользователями и баннерами.
7. Если данных больше `max_samples`, делает балансировку:
   берется максимум половина позитивов и оставшееся место заполняется негативами.
8. Сортирует interactions по `event_date`.
9. Делит выборку по уникальным датам `event_date` на `train/valid/test`, чтобы одна и та же дата не попадала в разные сплиты.
10. Границы дат подбираются так, чтобы доли сплитов были как можно ближе к целевым `validation_ratio` и `test_ratio` по числу строк.

### Текущие размеры после подготовки

При дефолтном `max_samples=250_000`:

- всего после семплирования: `250_000`
- позитивных: `125_000`
- негативных: `125_000`
- `train`: `175_000`
- `valid`: `50_000`
- `test`: `25_000`

Границы временного сплита на текущих данных:

- `train`: `2026-01-01` -> `2026-03-05`
- `valid`: `2026-03-05` -> `2026-03-23`
- `test`: `2026-03-23` -> `2026-03-31`

## Архитектура модели

Модель описана в [twotower/core.py](/home/adam/projects/twotower-project/twotower/core.py:1), [twotower/user_tower.py](/home/adam/projects/twotower-project/twotower/user_tower.py:1) и [twotower/item_tower.py](/home/adam/projects/twotower-project/twotower/item_tower.py:1).

### User tower

```text
Embedding(user_id) -> Linear -> ReLU -> LayerNorm
```

### Item tower

```text
Embedding(banner_id) -> Linear -> ReLU -> LayerNorm
```

### Scoring

Эмбеддинги L2-нормализуются, затем score считается как скалярное произведение:

```text
dot(normalize(user_embedding), normalize(item_embedding))
```

### Loss и optimizer

- `BCEWithLogitsLoss`
- `Adam`

## Обучение, оценка и инференс

Класс `TwoTower` сейчас умеет:

- `fit(train_df, valid_df)` - обучение с логированием `train_loss`, `valid_loss`, `valid_accuracy`;
- `fit_from_interactions(interactions_df)` - подготовка и split interactions внутри пакета по `validation_ratio` и `test_ratio` из `TwoTowerConfig`;
- `evaluate(test_df)` - расчет `test_loss`, `test_accuracy`, `recall_at_k`;
- `predict(user_ids, item_ids=None, top_k=None)` - top-k рекомендации по всем доступным item embeddings;
- `save_model(path)` - сохранение чекпойнта;
- `load_model(path)` - восстановление модели и маппингов.

Текущее поведение после последних правок:

- дополнительное семплирование и балансировка внутри `twotower` применяются только к train-части;
- `valid` и `test` больше не пересэмплируются внутри модели, чтобы метрики считались на реальном сплите;
- `predict(item_ids=...)` корректно игнорирует неизвестные `banner_id` без смещения индексов в ответе;
- `predict()` кэширует эмбеддинги полного каталога товаров внутри модели и переиспользует их между вызовами;
- `device=None` теперь безопасно выбирает доступное устройство автоматически;
- загрузка чекпойнта сначала идет на CPU, а затем модель переносится на итоговое устройство;
- `evaluate()` теперь возвращает еще и статистику отфильтрованных строк теста;
- `recall_at_k` не рекомендует баннеры, уже встречавшиеся пользователю в `train/valid`;
- поля `validation_ratio` и `test_ratio` теперь используются в `fit_from_interactions()`.
- `fit_from_interactions()` теперь строит train vocabulary без подглядывания в `test` и не делает sampling до temporal split.
- `fit_from_interactions()` теперь идет по более прямому пути: нормализация interactions -> split по датам -> `fit(train, valid)` -> подготовка test через train vocabulary.
- `split_interactions()` теперь делит данные именно по датам, а не по индексам строк, и подбирает temporal-границы ближе к целевым долям по числу строк.

### Что сохраняется в чекпойнт

- config модели;
- `state_dict`;
- `user_id_to_idx`;
- `item_id_to_idx`;
- `idx_to_user_id`;
- `idx_to_item_id`;
- `train_history`.

## Конфигурация

### Data config

Файл: [src/сonfig.py](/home/adam/projects/twotower-project/src/сonfig.py:1)

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

### Model config

Файл: [twotower/config.py](/home/adam/projects/twotower-project/twotower/config.py:1)

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

## Текущий сценарий запуска

Файл: [main.py](/home/adam/projects/twotower-project/main.py:1)

1. Создает `Config()`.
2. Загружает данные.
3. Строит ID-mapping из `users.csv` и `banners.csv`.
4. Подготавливает interactions и делает временной split.
5. Создает `TwoTower()` с дефолтным `TwoTowerConfig`.
6. Обучает модель через `fit(train_df, valid_df)`.
7. Оценивает модель на `test_df`.
8. Строит рекомендации для первых трех пользователей из `users.csv`.
9. Сохраняет модель в `artifacts/twotower_model.pth`.

Запуск:

```bash
python main.py
```

Через проектное окружение:

```bash
.venv/bin/python main.py
```

## Метрики

Во время работы проекта используются:

- `train_loss`
- `valid_loss`
- `valid_accuracy`
- `test_loss`
- `test_accuracy`
- `recall_at_k`

Особенность `recall_at_k`:

- считается только по пользователям, у которых в evaluation-части есть позитивные события;
- ограничивается первыми `max_eval_users`;
- для каждого пользователя исключает из кандидатов баннеры, уже виденные в `train/valid`.

Особенность `predict()`:

- при рекомендациях по полному каталогу эмбеддинги item tower переиспользуются из внутреннего кэша;
- кэш автоматически сбрасывается после `fit()` и `load_model()`.

Дополнительные поля из `evaluate()`:

- `test_input_rows`
- `test_rows_used`
- `test_rows_filtered`
- `test_unknown_user_rows`
- `test_unknown_item_rows`
- `test_rows_filtered_ratio`

## Ограничения текущей реализации

Сейчас это именно baseline, а не production-ready recommender.

- Несмотря на богатые таблицы `users.csv` и `banners.csv`, модель использует только `user_id` и `banner_id`.
- Все остальные признаки пока игнорируются.
- Cold-start для новых пользователей и баннеров не поддержан.
- Неизвестные `user_id` и `banner_id` по-прежнему отфильтровываются при подготовке и оценке, но теперь это явно отражается в метриках evaluation.
- При train-only vocabulary часть строк в `valid/test` может быть отброшена как unseen, и это теперь сознательное поведение ради более честной temporal оценки.
- `main.py` все еще использует внешний pipeline из `src/data.py`, хотя `twotower` уже умеет делать split самостоятельно через `fit_from_interactions()`.
- В проекте нет тестов.
- В `src` файл конфига называется `сonfig.py` с кириллической `с`, что легко пропустить при импортах и рефакторинге.

## Текущее состояние проекта

Состояние на `2026-04-18`:

- проект запускается как локальный учебный pipeline;
- базовый чекпойнт уже лежит в `artifacts/`;
- код организован просто и читаемо, но еще не выделен в устойчивую библиотечную структуру;
- в `twotower` исправлены безопасная загрузка модели, обработка `device`, честная evaluation без скрытого пересэмплирования, корректный `predict()` для частично невалидного `item_ids`, кэш item embeddings для инференса, общий data-prep модуль, более честный `recall_at_k` и train-only vocabulary в `fit_from_interactions()`;
- основной следующий шаг для качества модели - подключение реальных user/item features вместо чистых ID;
- основной следующий шаг для качества кода - убрать дублирование pipeline и покрыть базовые сценарии тестами.

## Установка

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
