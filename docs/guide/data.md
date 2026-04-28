# Data Preparation

## Expected format

Three CSV files are needed: **interactions**, **users** (optional), and **items** (optional).

### Interactions

The interactions table records every user–item event:

| Column | Type | Description |
|---|---|---|
| `event_date` | date / datetime | When the event occurred |
| `user_id` | int | User identifier |
| `item_id` | int | Item identifier (configurable) |
| `clicks` | int | Number of clicks (> 0 → positive label) |

The column names for user and item are configurable via `user_col` / `item_col` when constructing `TwoTower`. The `event_date` column is always required by name.

### Users and items (side features)

Side-feature tables are flat CSVs with one row per entity:

```
user_id, age_group, country
1001, 25-34, US
1002, 18-24, DE
```

All columns are treated as categorical strings. See [Training → Side features](training.md#side-features) for how to declare them.

## Splitting by date

`split_interactions` performs a **temporal** split — it never leaks future events into the training set:

```python
from twotower import split_interactions

train_df, valid_df, test_df = split_interactions(
    interactions_df,
    validation_ratio=0.2,
    test_ratio=0.1,
)
```

The function finds the date boundaries that best match the requested ratios, keeping all interactions from a given date in the same split. It requires at least 3 unique event dates.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `validation_ratio` | `0.2` | Fraction of interactions for validation |
| `test_ratio` | `0.1` | Fraction of interactions for test |

### Returns

A 3-tuple `(train_df, valid_df, test_df)` — each is a slice of the original DataFrame sorted by `event_date`.

## Loading data

A typical loading pattern:

```python
import pandas as pd
from twotower import split_interactions

interactions = pd.read_csv("interactions.csv")
users = pd.read_csv("users.csv")
items = pd.read_csv("items.csv")

train_df, valid_df, test_df = split_interactions(interactions)
```

Then pass the splits directly to `TwoTower.fit`:

```python
X_train = train_df.drop(columns=["clicks"])
y_train = train_df["clicks"]

X_valid = valid_df.drop(columns=["clicks"])
y_valid = valid_df["clicks"]
```
