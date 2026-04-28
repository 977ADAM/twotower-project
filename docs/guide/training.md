# Training

## Basic training

```python
from twotower import TwoTower

model = TwoTower(
    epochs=25,
    batch_size=2048,
    tower_dims=(256, 128),
)

history = model.fit(
    X_train=X_train, y_train=y_train,
    X_valid=X_valid, y_valid=y_valid,
)
```

`fit` returns the training history — a list of dicts with per-epoch metrics (train loss, valid loss, recall@k).

## Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `user_col` | `"user_id"` | Column name for user IDs in your DataFrame |
| `item_col` | `"banner_id"` | Column name for item IDs in your DataFrame |
| `tower_dims` | `(256, 128)` | MLP hidden layer sizes for both towers. Empty tuple `()` for a linear projection. |
| `hidden_dim` | `64` | Output embedding dimension |
| `dropout` | `0.0` | Dropout rate applied inside the MLP towers |
| `retrieval_temperature` | `0.1` | Softmax temperature for in-batch InfoNCE loss |
| `learning_rate` | `1e-3` | Adam optimizer learning rate |
| `batch_size` | `2048` | Training mini-batch size |
| `epochs` | `25` | Maximum number of training epochs |
| `max_samples` | `250_000` | Cap on the number of positive training pairs; `None` to use all |
| `seed` | `42` | Random seed for reproducibility |
| `device` | `"cpu"` | `"cpu"`, `"cuda"`, or `None` (auto-detect) |

## Tower architecture

Each tower follows the same architecture:

```
ID embedding  ──┐
side features ──┴─▶  concat  ──▶  MLP  ──▶  LayerNorm  ──▶  embedding
```

`tower_dims` controls the MLP hidden layers. Each hidden layer applies:

1. `Linear`
2. `BatchNorm1d`
3. `ReLU`
4. `Dropout` (if `dropout > 0`)

The final linear projects to `hidden_dim`, followed by `LayerNorm`.

## Negative sampling

Control how negative examples are constructed during training:

```python
from twotower import NegativeSampling

model.fit(
    ...,
    negative_sampling=NegativeSampling(
        observed_ratio=0.8,       # 80 % from observed non-positive interactions
        in_batch_loss_weight=0.5, # add InfoNCE in-batch contrastive loss
    ),
)
```

| Parameter | Default | Description |
|---|---|---|
| `observed_ratio` | `0.8` | Fraction of negatives drawn from observed (non-positive) items vs. random items |
| `in_batch_loss_weight` | `0.0` | Weight for the InfoNCE in-batch contrastive loss; `0.0` disables it |

## Early stopping

Early stopping is enabled by default and monitors validation loss:

```python
from twotower import EarlyStopping

model.fit(
    ...,
    early_stopping=EarlyStopping(
        patience=5,
        metric="valid_loss",  # or e.g. "recall_at_100"
        min_delta=1e-4,
    ),
)
```

Pass `early_stopping=None` to train for the full `epochs`.

| Parameter | Default | Description |
|---|---|---|
| `patience` | `5` | Stop after this many epochs without improvement |
| `metric` | `"valid_loss"` | Metric to monitor; loss metrics minimise, `recall_at_*` maximise |
| `min_delta` | `1e-4` | Minimum improvement to count as progress |

## Side features

Pass side-feature DataFrames and declare which columns to use:

```python
from twotower import TwoTower, FeatureConfig, MultiFeatureSpec

model = TwoTower()

model.fit(
    X_train=X_train, y_train=y_train,
    X_valid=X_valid, y_valid=y_valid,
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

`users_df` and `items_df` must be provided together. They must contain an ID column matching `user_col` / `item_col`.

### FeatureConfig

| Parameter | Description |
|---|---|
| `scalar_features` | Tuple of column names, each encoded as a single embedding |
| `multi_features` | Tuple of `MultiFeatureSpec`, each pooled across its columns |

### MultiFeatureSpec

| Parameter | Description |
|---|---|
| `name` | Feature name (used for embedding tables) |
| `columns` | Tuple of column names holding the values for this feature |
