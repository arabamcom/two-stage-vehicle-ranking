# Two-Stage Personalized Vehicle Ranking in Online Auctions

Reference implementation of the two-stage framework described in *Two-Stage Personalized Vehicle
Ranking in Online Auctions Using Heterogeneous Graph Neural Networks and LambdaRank*.

1. **Representation learning.** A heterogeneous GNN learns user and vehicle embeddings on a
   user-vehicle-property graph with bid, purchase, favorite, alert and vehicle-property relations.
   It is trained with a symmetric InfoNCE objective on bid edges.
2. **Learning to rank.** A LightGBM LambdaRank model ranks the vehicles of an auction day's roster
   for each user. It combines graph-embedding similarity with user-history, vehicle,
   user-vehicle compatibility and daily-roster features.

The original user and auction records are proprietary and are not released. The repository
contains a synthetic data generator that writes data with the same graph schema, node and
relation types, and file formats. With it, the complete pipeline runs end to end. Synthetic data
does not reproduce the statistical properties of the proprietary records, so the repository
verifies the implementation; it does not reproduce the reported numbers, and results on it are
not evidence for or against any model component.

## Installation

Python 3.11 was used. Install PyTorch and the PyTorch Geometric companion wheels for your
platform first (see `requirements.txt`), then:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
python scripts/run_all.py                 # about 3 to 6 minutes on a single consumer GPU
python scripts/run_all.py --max-epochs 3  # shorter GNN training for a smoke run
pytest
```

| Step | Script | Output |
|---|---|---|
| Synthetic export | `scripts/generate_synthetic_data.py` | `data/synthetic/` |
| GNN training and embedding export | `scripts/train_gnn.py` | `runs/gnn/` |
| Ranking dataset and evaluation tasks | `scripts/build_ranking_dataset.py` | `runs/ranking_data/` |
| LambdaRank, full model and no-GNN ablation | `scripts/train_ranker.py` | `runs/ranker_full/`, `runs/ranker_no_gnn/` |
| Full-roster evaluation | `scripts/evaluate.py` | `runs/evaluation.json` |

## Method

### Time windows
The cutoffs in `cutoff_spec.json` are `T_train < T_val < T_test`, and `T_emb = T_train - W` days
with `W = 180`.

| Window | Used for | Everything a row may use is taken as of |
|---|---|---|
| up to `T_emb` | GNN training | `T_emb` |
| `(T_emb, T_emb + 17d]`, then 15 more days | GNN validation and link-prediction test | `T_emb` |
| `(T_emb, T_train]` | LambdaRank training positives | `T_emb` |
| `(T_train, T_val]` | LambdaRank validation | `T_train` |
| `(T_val, T_test - 7d]` | LambdaRank test | `T_train` |
| `(T_test - 7d, T_test]` | Locked holdout, evaluation only | `T_train` |

"Everything" covers user history and preference features, categorical affinities, the
message-passing graph, node-feature scalers and graph embeddings. No row therefore sees its own
target interaction, and training and evaluation rows follow the same rule. A row's auction day is
the listing day of its vehicle in the decision-time roster (`roster.parquet`).

### Graph (`graph/`)
- **Nodes:** `user`, `item` (vehicle) and `property`.
- **Relations:** `BID`, `PURCHASE`, `FAVORITE`, `ALERT`, and `HAS_PROPERTY` from a vehicle to its
  property values. Every relation has a reverse, so users receive messages from all four behaviors.
- **Edge weights:**
  - bid: `3 * (1 + 0.2 * ln(1 + number of bids))`;
  - purchase 10, favorite 2, alert 1;
  - each is multiplied by a relation weight (`HAS_PROPERTY` 0.8, all others 1).
- **Property values:**
  - brand, model family, fuel type, transmission and vehicle type;
  - five quintile buckets of the real (deflated) price;
  - age buckets at 3, 6 and 10 years;
  - a missing-value node for a missing price or age.

  Price cut points, the kept property vocabulary and the price-index reference medians use only
  vehicles listed up to `T_emb`, the earliest cutoff of the pipeline (in the synthetic generator,
  `static_cutoff_days` must therefore be at least `W`). Brand and model-family values
  below 20 vehicles, or outside 99% coverage, are pruned. Prices are deflated with each month's
  median listing price.
- **Time-sliced graph at a cutoff:** keeps every timestamped edge up to the cutoff. An edge
  without a timestamp is kept when its source node is visible, so property-to-vehicle messages
  reach newly listed vehicles.

### GNN (`gnn/`)
- **Encoder:** type-specific projections into 256 dimensions, a learnable 32-dimensional embedding
  for property nodes, two `GraphConv` layers per relation averaged per node type, residual
  connections, dropout 0.3, and 128-dimensional L2-normalized outputs.
- **Disjoint targets and messages:** in every epoch, 30% of the vehicles with training bids become
  targets. Their bids are the InfoNCE positives, and all of their behavior edges are removed from
  the message-passing graph. A target vehicle therefore looks exactly like a newly listed auction
  vehicle, while users keep collaborative messages from their other vehicles.
- **Loss:** symmetric InfoNCE with temperature 0.15 over in-batch negatives. Other known
  positives of the same user are masked.
- **Selection and link-prediction test:** selection uses the validation InfoNCE loss. Link
  prediction is tested on later bids, each with three negatives drawn from the roster of the same
  auction day.
- **Embedding export:** the same weights are run twice, at `T_emb` for training rows and at
  `T_train` for evaluation rows.

### Ranking dataset (`ranking/`)
- **Relevance labels:** 0 for a sampled non-interaction, then alert 1, favorite 2, bid 5 and
  purchase 6. A pair with several behaviors keeps its highest label.
- **Positive cap:** at most 150 training positives per user, sampled by label priority. Training
  rows are weighted by the recency of the user's last training positive (half-life 180 days,
  floor 0.1).
- **Sampled non-interactions:** for a user-day with `n` positives, `4n` distinct vehicles are drawn
  from that day's roster, excluding every vehicle the user interacted with up to the split cutoff.
  Adjacent days are added when the roster has fewer than 20 vehicles. These rows are reference
  non-interactions, not verified dislikes.
- **Query groups:** user-day by default; user-only grouping is available. Rows are shuffled within
  groups so that ties never favor positives.
- **Features:** 50 in total:
  - 13 user-history features;
  - 9 price, age and damage-record features plus 8 appraisal features of the vehicle;
  - 7 user-vehicle fit features;
  - 6 roster-position features;
  - 5 categorical affinities (share of the user's label-weighted behavior on the candidate's
    value);
  - 2 embedding features: cosine similarity and Euclidean distance. These are zero for users
    without behavior before the snapshot.

### Evaluation (`evaluation/`)
- **Sampled-list diagnostic:** `scripts/train_ranker.py` reports it on the test split, with
  analytic random controls.
- **Full-roster evaluation:** `scripts/evaluate.py` ranks every holdout positive against all
  vehicles of its day's roster.
  - **Candidates:** the user's other positives of the window and earlier interactions are
    removed, so a perfect ranker reaches NDCG 1.
  - **Ties:** NDCG@10 and Hit@10 are tie-aware, that is, taken in expectation over a random order
    of tied candidates.
  - **Systems:** the full model, the no-GNN ablation, content affinity (a missing value ranks
    lowest), a model-family popularity rate (bids per listed vehicle in the previous 90 days) and
    random ordering.
  - **Uncertainty:** paired differences come with 95% intervals from 2,000 user-clustered
    bootstrap resamples.
- **Segments:** results are reported for ranking-warm users, who have LambdaRank training
  positives, and ranking-cold users, who do not.

## Using your own data

Write an export directory with these files. Every index is contiguous from 0; every timestamp is
in UTC epoch seconds.

| File | Columns |
|---|---|
| `nodes_users.parquet` | `u_idx` and static numeric user attributes (the synthetic data has `has_plan`); history features are computed by the pipeline |
| `nodes_item.parquet` | `i_idx`, `item_price_real_log1p`, `item_price_index_log1p`, `item_price_index_brand_log1p`, `item_age_years`, `item_age_log1p`, `damage_record_log1p`, `damage_record_real_log1p`, `damage_record_to_price`, `damage_record_has_value`, `appraisal_severity_mean` and further `appraisal_*` columns |
| `nodes_item_props.parquet` | `p_key` (`Type::value`, with types `Brand`, `ModelFamily`, `FuelType`, `Transmission`, `VehicleType`, `PriceBucket`, `AgeBucket`), `p_idx` |
| `edges_item_prop.parquet` | `i_idx`, `p_idx` |
| `edges_{bid,purchase,fav,alert}.parquet` | `u_idx`, `i_idx`, `ts` (one row per pair, latest time); bid also has `count` |
| `roster.parquet` | `epoch_day`, `i_idx`: the vehicles listed in each day's auction |
| `cutoff_spec.json` | `train_cutoff_epoch`, `val_cutoff_epoch`, `test_cutoff_epoch` |

`build_property_graph` in `graph/properties.py` builds the two property files from a vehicle table.

## Configuration

`configs/gnn.yaml` and `configs/ranking.yaml` hold all hyperparameters; `configs/synthetic.yaml`
controls the generator. All stochastic steps use seed 42; on a GPU, floating-point
nondeterminism in scatter operations can still cause small run-to-run differences.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
