# Prediction

## predict

```python
recommendations = model.predict(
    user_ids=[1001, 1002, 1003],
    top_k=10,
)
```

Returns a dict mapping each user ID to their ranked list of recommendations:

```python
{
    1001: [{"item_id": 42, "score": 0.97}, {"item_id": 7, "score": 0.94}, ...],
    1002: [...],
}
```

The item key in each dict matches the `item_col` used when constructing the model (defaults to `"banner_id"`).

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `user_ids` | `None` | User IDs to predict for. If `None`, up to 10 known users are used. |
| `item_ids` | `None` | Candidate item IDs. If `None`, all known items are candidates. |
| `top_k` | `None` | Number of items to return per user. Falls back to the model's `top_k` config value. |
| `exclude_seen` | `True` | Whether to exclude items the user interacted with in training. |
| `strict` | `False` | If `True`, raises an error for unknown user or item IDs. If `False`, they are silently skipped. |

## Restricting the candidate pool

Pass `item_ids` to predict from a subset of items:

```python
# Only consider items in a specific category
candidate_ids = items_df.loc[items_df["category"] == "electronics", "item_id"].tolist()
recommendations = model.predict(user_ids=[1001], item_ids=candidate_ids, top_k=5)
```

## Saving and loading

A fitted model can be persisted and reloaded:

```python
# Save
model.save_model("model.pth")

# Load into a fresh instance
model2 = TwoTower()
model2.load_model("model.pth")

# Or use method chaining
model2 = TwoTower().load_model("model.pth")

recommendations = model2.predict(user_ids=[1001])
```

The checkpoint stores the model weights, config, ID mappings, training history, and side-feature metadata. Side-feature DataFrames are not saved; the embeddings baked into the tower weights are preserved.
