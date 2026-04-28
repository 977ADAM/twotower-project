# Evaluation

## evaluate

```python
metrics = model.evaluate(X_test)
```

`evaluate` computes retrieval metrics on a held-out test set. `X_test` must contain the same user and item ID columns used during training, plus either a `clicks` or `label` column.

```python
# With a clicks column — label is derived as (clicks > 0)
metrics = model.evaluate(test_df)

# With an explicit label column (0.0 / 1.0 floats)
metrics = model.evaluate(test_df.rename(columns={"clicks": "label"}))

# Evaluate at a custom top-k
metrics = model.evaluate(test_df, top_k=50)
```

### Return value

A dict with the following keys (top-k values come from `eval_top_ks` plus `top_k`):

```python
{
    "recall_at_50":  0.34,
    "recall_at_100": 0.41,
    "popularity_recall_at_50": 0.28,   # baseline: recommend most popular items
    "popularity_recall_at_100": 0.33,
}
```

## Recall@k

Recall@k measures the fraction of relevant items that appear in the top-k recommendations, averaged over users:

$$\text{Recall@k} = \frac{1}{|U|} \sum_{u \in U} \frac{|\text{recommended}_u \cap \text{relevant}_u|}{|\text{relevant}_u|}$$

Seen items (interactions from the training set) are excluded from the recommendation list by default.

## Popularity baseline

`popularity_recall_at_k` is a naïve baseline that recommends the globally most popular training items to every user (seen items excluded). Values above the baseline indicate the model has learned personalised signal beyond simple popularity.

## Configuring evaluation

These parameters on `TwoTower` control how evaluation is performed:

| Parameter | Default | Description |
|---|---|---|
| `eval_top_ks` | `(50, 100, 300)` | Top-k values to evaluate during training and in `evaluate()` |
| `max_eval_users` | `500` | Maximum number of users sampled for evaluation (for speed) |
| `eval_during_training` | `True` | Whether to compute recall metrics after each training epoch |

## Training history

After `fit`, the full per-epoch history is available:

```python
history = model.fit(...)

for epoch in history:
    print(epoch)
# {'epoch': 1, 'train_loss': 0.62, 'valid_loss': 0.58, 'recall_at_100': 0.12, ...}
```
