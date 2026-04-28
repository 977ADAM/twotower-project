# twotower

A two-tower retrieval model library for recommendation systems, built on PyTorch.

Train a user tower and item tower jointly with BPR loss, then retrieve top-k items for any user in milliseconds via dot-product search.

## Installation

```bash
pip install twotower
```

Requires Python 3.11+ and PyTorch 2.11+.

## Quick start

```python
import pandas as pd
from twotower import TwoTower, TwoTowerConfig

# X must have user_id and banner_id columns; y is a binary label (1 = positive interaction)
model = TwoTower()

history = model.fit(
    X_train=train_df[["user_id", "banner_id"]],
    y_train=train_df["label"],
    X_valid=valid_df[["user_id", "banner_id"]],
    y_valid=valid_df["label"],
)

predictions = model.predict(user_ids=[1, 2, 3], top_k=10)
# {1: [{"banner_id": 42, "score": 0.94}, ...], ...}

metrics = model.evaluate(test_df)
# {"recall_at_100": 0.41, "popularity_recall_at_100": 0.45, ...}

model.save_model("model.pth")
model = TwoTower().load_model("model.pth")
```

## Configuration

```python
from twotower import TwoTower, TwoTowerConfig

model = TwoTower(TwoTowerConfig(
    user_embedding_dim=64,   # user tower output dim
    item_embedding_dim=64,   # item tower output dim
    hidden_dim=64,           # projection layer width
    learning_rate=1e-3,
    batch_size=2048,
    epochs=25,
    top_k=100,               # default k for predict/evaluate
    eval_top_ks=(50, 100, 300),  # k values logged during training
    max_samples=250_000,     # cap training interactions
    device="cpu",            # "cpu" | "cuda" | None (auto)
    seed=42,
))
```

## Side features

Pass categorical features for users and items to improve cold-start performance.

```python
from twotower import FeatureConfig, MultiFeatureSpec

user_feature_config = FeatureConfig(
    scalar_features=("age_bucket", "gender", "city"),
    multi_features=(
        MultiFeatureSpec("interests", columns=("interest_1", "interest_2", "interest_3")),
    ),
)
item_feature_config = FeatureConfig(
    scalar_features=("category", "brand", "target_gender"),
)

model.fit(
    X_train=..., y_train=...,
    X_valid=..., y_valid=...,
    users_df=users_df,        # must contain user_id + feature columns
    items_df=items_df,        # must contain banner_id + feature columns
    user_feature_config=user_feature_config,
    item_feature_config=item_feature_config,
)
```

All feature values are treated as categorical strings. Numeric values should be bucketed before passing.

## Training options

### Negative sampling

```python
from twotower import NegativeSampling

model.fit(
    ...,
    negative_sampling=NegativeSampling(
        observed_ratio=0.8,       # fraction of negatives from observed non-clicks
        in_batch_loss_weight=0.0, # add InfoNCE in-batch contrastive loss (0 = disabled)
    ),
)
```

### Early stopping

```python
from twotower import EarlyStopping

model.fit(
    ...,
    early_stopping=EarlyStopping(
        metric="recall_at_100",  # or "valid_loss"
        patience=5,
        min_delta=1e-4,
    ),
    # pass early_stopping=None to train for all epochs
)
```

## API reference

### `TwoTower.fit`

```python
model.fit(
    X_train, y_train,           # interactions DataFrame + binary labels
    X_valid, y_valid,
    users_df=None,              # optional side features for users
    items_df=None,              # optional side features for items
    user_feature_config=None,
    item_feature_config=None,
    negative_sampling=NegativeSampling(),
    early_stopping=EarlyStopping(),
) -> list[dict[str, float]]     # training history
```

### `TwoTower.predict`

```python
model.predict(
    user_ids=None,       # list[int] | None — defaults to first 10 known users
    item_ids=None,       # list[int] | None — defaults to all known items
    top_k=None,          # int | None — defaults to TwoTowerConfig.top_k
    exclude_seen=True,   # exclude items the user interacted with during training
    strict=False,        # raise ValueError for unknown IDs instead of skipping
) -> dict[int, list[dict[str, float]]]  # {user_id: [{"banner_id": ..., "score": ...}]}
```

### `TwoTower.evaluate`

```python
model.evaluate(
    X_test,      # DataFrame with user_id, banner_id, and label (or clicks) columns
    top_k=None,
) -> dict[str, float]   # recall_at_k, popularity_recall_at_k, test_loss, ...
```

### `TwoTower.save_model` / `TwoTower.load_model`

```python
model.save_model("path/to/model.pth")
model = TwoTower().load_model("path/to/model.pth")
```

## Data format

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | User identifier |
| `banner_id` | int | Item identifier |
| `label` | float | 1.0 = positive interaction, 0.0 = negative |
| `clicks` | int | Alternative to `label` — any value > 0 is treated as positive |
| `event_date` | date | Optional — used for chronological train/valid/test splits |

## License

MIT
