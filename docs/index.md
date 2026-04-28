# twotower

A two-tower retrieval model for recommendations, with a scikit-learn–style API.

## Installation

```bash
pip install twotower
```

For the optional FastAPI / Gradio serving layer:

```bash
pip install "twotower[app]"
```

## Quickstart

```python
import pandas as pd
from twotower import TwoTower, split_interactions

# Load your interactions (must have event_date, user_id, item_id, clicks columns)
interactions = pd.read_csv("interactions.csv")

# Split by date into train / validation / test
train_df, valid_df, test_df = split_interactions(interactions)

# Separate features from labels
X_train, y_train = train_df.drop(columns=["clicks"]), train_df["clicks"]
X_valid, y_valid = valid_df.drop(columns=["clicks"]), valid_df["clicks"]

# Train
model = TwoTower(epochs=10)
model.fit(
    X_train=X_train, y_train=y_train,
    X_valid=X_valid, y_valid=y_valid,
)

# Predict top-10 items for a list of users
recommendations = model.predict(user_ids=[1, 2, 3], top_k=10)

# Evaluate recall on held-out test data
metrics = model.evaluate(test_df)

# Persist
model.save_model("model.pth")
model.load_model("model.pth")
```

## Design

**twotower** follows the *fit → predict → evaluate* pattern familiar from scikit-learn.

- **User tower** and **item tower** each map an entity ID (plus optional side features) through an MLP to a shared embedding space.
- At inference time, user and item embeddings are compared with a dot product; the top-k items are returned per user.
- Training uses Bayesian Personalised Ranking (BPR) loss with optional in-batch InfoNCE contrastive loss.

## What's in the box

| Symbol | Purpose |
|---|---|
| `TwoTower` | Main model class |
| `split_interactions` | Temporal train / valid / test split |
| `FeatureConfig` | Declare side features for users or items |
| `MultiFeatureSpec` | Describe a multi-valued side feature |
| `NegativeSampling` | Control negative sampling strategy |
| `EarlyStopping` | Configure early stopping |
