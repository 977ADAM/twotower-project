# twotower

A two-tower retrieval model for recommendations, built on PyTorch.

Train a user tower and an item tower jointly, then retrieve top-k items for any user via dot-product search.

## Installation

```bash
pip install twotower
```

Requires Python 3.11+ and PyTorch 2.11+.

## Quick start

```python
import pandas as pd
from twotower import TwoTower, split_interactions

interactions = pd.read_csv("interactions.csv")  # needs event_date, user_id, item_id, clicks

train_df, valid_df, test_df = split_interactions(interactions)

model = TwoTower(epochs=25, tower_dims=(256, 128))
model.fit(
    X_train=train_df.drop(columns=["clicks"]),
    y_train=train_df["clicks"],
    X_valid=valid_df.drop(columns=["clicks"]),
    y_valid=valid_df["clicks"],
)

recommendations = model.predict(user_ids=[1, 2, 3], top_k=10)
# {1: [{"item_id": 42, "score": 0.97}, ...], ...}

metrics = model.evaluate(test_df)
# {"recall_at_100": 0.41, "popularity_recall_at_100": 0.33, ...}

model.save_model("model.pth")
model = TwoTower().load_model("model.pth")
```

## Data format

`split_interactions` expects a DataFrame with at least these columns:

| Column | Type | Description |
|--------|------|-------------|
| `event_date` | date / datetime | Used for chronological splitting |
| `user_id` | int | User identifier (configurable via `user_col`) |
| `item_id` | int | Item identifier (configurable via `item_col`) |
| `clicks` | int | `> 0` → positive label |

Column names for users and items are configurable:

```python
model = TwoTower(user_col="user_id", item_col="product_id")
```

## Key hyperparameters

```python
model = TwoTower(
    tower_dims=(256, 128),   # MLP hidden layers inside each tower; () for linear
    hidden_dim=64,           # output embedding dimension
    dropout=0.0,             # dropout rate in MLP towers
    learning_rate=1e-3,
    batch_size=2048,
    epochs=25,
    max_samples=250_000,     # cap on positive training pairs; None = use all
    eval_top_ks=(50, 100, 300),
    top_k=100,               # default k for predict / evaluate
    device="cpu",            # "cpu" | "cuda" | None (auto-detect)
    seed=42,
)
```

## Side features

Pass categorical features for users and items to improve cold-start performance.
All values are treated as strings — bucket numerics before passing.

```python
from twotower import FeatureConfig, MultiFeatureSpec

model.fit(
    X_train=..., y_train=...,
    X_valid=..., y_valid=...,
    users_df=users_df,
    items_df=items_df,
    user_feature_config=FeatureConfig(
        scalar_features=("country", "age_group"),
    ),
    item_feature_config=FeatureConfig(
        scalar_features=("category",),
        multi_features=(
            MultiFeatureSpec(name="tags", columns=("tag_1", "tag_2", "tag_3")),
        ),
    ),
)
```

## Negative sampling

```python
from twotower import NegativeSampling

model.fit(
    ...,
    negative_sampling=NegativeSampling(
        observed_ratio=0.8,        # fraction of negatives from observed non-clicks
        in_batch_loss_weight=0.5,  # add InfoNCE contrastive loss; 0.0 = BPR only
    ),
)
```

## Early stopping

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

## Documentation

Full documentation at **[977adam.github.io/twotower](https://977adam.github.io/twotower)**.

## License

MIT
