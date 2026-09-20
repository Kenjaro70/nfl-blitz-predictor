# NFL Blitz Predictor

Pre-snap binary classifier: given the offensive context (down, distance, formation, personnel, score, clock), predict whether the defense will blitz (5+ pass rushers).

> **Honest baseline:** ROC AUC **0.708 ± 0.018** across 10 by-game splits (0.500 = no skill); the shipped model scores 0.682 on its own 25 held-out games. The shipped model is a **logistic regression** — it ties a random forest on AUC, so the tie went to the model you can read ([why](#why-logistic-regression)). Probabilities are isotonic-calibrated, and the alert threshold is derived from an explicit cost ratio rather than left at 0.5. Blitzes are ~25% of plays, so accuracy is the wrong headline — see [Why not accuracy?](#why-not-accuracy). Trained on 2021 NFL weeks 1–8 from the Big Data Bowl 2023 dataset. Full writeup in [STORY.md](STORY.md).

## Results

Single held-out split (25 games, 1,750 plays, 26.1% blitz rate), reported at the
tuned decision threshold of 0.25:

| Model | ROC AUC | Accuracy | Blitz precision | Blitz recall |
|---|---|---|---|---|
| Always predict "no blitz" | 0.500 | **0.739** | — | 0.00 |
| Random forest (challenger) | 0.689 | 0.619 | 0.37 | 0.66 |
| **Logistic regression (shipped)** | **0.682** | 0.634 | 0.38 | 0.64 |

One split of 122 games is a noisy measurement, so the training run repeats the whole
thing over 10 seeds:

| Metric | Mean ± SD over 10 by-game splits |
|---|---|
| **Logistic regression AUC (shipped)** | **0.708 ± 0.018** |
| Random forest AUC (challenger) | 0.711 ± 0.017 |
| Logistic regression accuracy | 0.657 ± 0.029 |
| Majority-class accuracy | 0.755 ± 0.018 |

The spread matters: a single split can land anywhere from 0.688 to 0.741 on identical
code. Any claim resting on a difference smaller than ~0.02 AUC is not measurable here.

### Why logistic regression

The random forest beats it by **+0.0032 AUC, standard error 0.0033**, winning 6 of 10
splits. That is a tie, not a win. When two models are indistinguishable the tie goes to
the one you can explain, and the other differences are not close:

| | Logistic regression | Random forest |
|---|---|---|
| Artifact size | **6.9 KB** | 7.1 MB |
| Explanation | 14 numeric coefficients + per-level effects | 300 trees |
| Unseen category | all-zero block (`handle_unknown='ignore'`) | arbitrary split on a `-1` sentinel |

The whole model is in `/model-info` under `coefficients`. Numeric features were
standardised, so coefficients are per standard deviation and directly comparable:

| Feature | Coefficient | Odds ratio |
|---|---|---|
| `defendersInBox` | +0.642 | **1.90** |
| `down` | +0.200 | 1.22 |
| `yardsToGo` | −0.173 | 0.84 |
| `score_differential` | +0.168 | 1.18 |
| `quarter` | +0.142 | 1.15 |
| `yards_to_goal` | −0.141 | 0.87 |

Every sign is football-sensible: more defenders near the line and later downs push blitz
probability up, longer distance-to-go and being further from the opponent's goal push it
down. One standard deviation more in the box multiplies the odds of a blitz by ~1.9.

**Two costs of the swap, stated plainly:**

1. **Predictions are less extreme at the tails.** A linear model can't represent "9 in
   the box *at the goal line*" as an interaction. That scenario reads 0.356 (1.45x base
   rate) where the forest read 0.517 (2.10x). Ordering is preserved; the spread is
   compressed.
2. **Large per-team coefficients on 8 weeks of data.** `defensiveTeam=LV` lands at
   −1.13 and `defensiveTeam=LA` at +0.92. Those may be real coaching tendencies, but at
   ~4 games per team they're the most likely thing here to be overfit — and a linear
   model makes them visible in a way the forest did not.

The random forest is still trained on every run as the challenger, so "it's a tie" stays
checkable rather than asserted — see `baseline_comparison` in `/model-info`.

### Why not accuracy?

Because the majority-class model beats us on it — 0.739 vs. 0.634 — and it has no
predictive value whatsoever. Any classifier on a 26% base rate can score 74% by
refusing to ever predict the positive class.

Accuracy drops further once the threshold is tuned, and that's the intended trade: the
model is an **alert**, and a missed blitz costs more than a false alarm. At the tuned
threshold it catches **64% of actual blitzes**, and when it fires it's right **38%** of
the time against a 26.1% base rate. See [The decision rule](#the-decision-rule).

## The decision rule

A probability is not a decision. The consumer here is an offensive analyst deciding
whether to keep extra protection in, and the two mistakes are not equally bad:

- **Missed blitz** (we say no, defense blitzes) → an unblocked rusher with a free run at
  the QB. Sack or hurry.
- **False alarm** (we say blitz, defense rushes four) → a back or TE stays in to block
  instead of running a route. One fewer target, mild EPA cost.

This project assumes a missed blitz is **3× worse** than a false alarm. That is an
assumption, not a measurement — I tried to derive it from EPA and couldn't (see
[below](#can-the-cost-ratio-be-measured)). So it's a named constant in `train_model.py`,
its provenance is recorded in `model_metadata.json`, and the training run prints how the
operating point moves if you disagree:

| FN:FP cost | Threshold | Blitz precision | Blitz recall | Alert rate |
|---|---|---|---|---|
| 1:1 | 0.50 | 0.63 | 0.09 | 0.04 |
| 2:1 | 0.33 | 0.50 | 0.33 | 0.17 |
| **3:1 (shipped)** | **0.25** | **0.38** | **0.64** | **0.44** |
| 5:1 | 0.17 | 0.32 | 0.81 | 0.66 |
| 8:1 | 0.11 | 0.29 | 0.90 | 0.81 |

For calibrated probabilities the cost-optimal cutoff has a closed form —
`C_FP / (C_FP + C_FN)`, so 1/(1+3) = **0.25**. A grid search over out-of-fold training
predictions independently picked **0.22** — close enough to confirm the calibration is
doing its job, and the small gap is why the training run warns if the two ever diverge by
more than 0.05.

The threshold is chosen on **out-of-fold training predictions only** (`GroupKFold`
within the training games). Tuning it on the test set would be the same mistake as
tuning a hyperparameter there.

The tuned threshold is slightly cheaper than the default here — 0.5554 vs. 0.5571 cost
per play — but the gap is small enough to call a wash. The point isn't the saving. It's
that the operating point is now a stated choice that moves correctly when the cost
assumption changes, instead of an accident of sklearn's default.

### Can the cost ratio be measured?

I tried. `training/analyze_cost_ratio.py` joins these plays to nflverse play-by-play for
real EPA (**100% join rate**, 8,557/8,557) and compares the four cells of {defense
blitzed} × {offense kept extra protection}. It does **not** identify the ratio: the
false-alarm cost comes out at −0.156 EPA, i.e. the wrong sign, because
`pff_role == 'Pass Block'` is assigned *post-snap* and "extra blocker" turns out to mark
max-protect play design (61% play-action, 12.9 air yards) rather than a protection call.

What it does establish:

| Quantity | Value |
|---|---|
| EPA cost of a sack | **−2.06** |
| Sack rate, blitz vs. 4-man rush | 8.8% vs. 6.1% |
| EPA via the sack channel | −0.055 |
| **Net EPA effect of a blitz** | **−0.036**, 95% CI [−0.094, +0.025] |

The net effect is smaller than the sack channel because a blitz also gives up coverage.
Blitzing was close to EPA-neutral in 2021 — which **bounds what this project can be worth**,
whatever its AUC. Full discussion in [STORY.md](STORY.md).

Identifying the ratio needs pre-snap protection *intent*, which means the unused
`week*.csv` tracking data rather than post-snap role labels.

## Calibration

`class_weight='balanced'` makes the raw probabilities unusable as probabilities. The
uncalibrated model predicted a mean blitz probability of **0.440** on a test set whose
actual rate was **0.261** — it was overstating blitz risk by roughly 2×.

Fixed with isotonic regression fit on out-of-fold training predictions:

| | Brier score | Mean predicted | Observed |
|---|---|---|---|
| Raw | 0.2148 | 0.440 | 0.261 |
| **Isotonic (shipped)** | **0.1766** | **0.234** | 0.261 |

17.8% better Brier. Isotonic is monotone, so AUC is unchanged — this fixes what the
numbers *mean*, not how well they rank. Reliability after calibration, from
`/model-info`:

| Predicted bucket | n | Mean predicted | Observed rate |
|---|---|---|---|
| 0.00–0.15 | 579 | 0.102 | 0.147 |
| 0.15–0.25 | 407 | 0.202 | 0.199 |
| 0.25–0.35 | 462 | 0.299 | 0.299 |
| 0.35–0.45 | 217 | 0.368 | 0.465 |
| 0.45–0.60 | 69 | 0.539 | 0.551 |
| 0.60–1.00 | 16 | 0.752 | 0.813 |

The 0.35–0.45 bucket is the one visibly off (0.368 predicted vs 0.465 observed, n=217).
Isotonic fixes the overall level, not every local wobble.

The API serves calibrated probabilities. `blitz_calibrator.joblib` is a separate
artifact so the two stages stay inspectable.

Built in the same train-once / serve-many pattern as `titanic-ml-project`:
- `training/` — Jupyter + full ML stack; produces `.joblib` artifacts.
- `serving/` — Slim FastAPI image; loads artifacts and exposes a REST API.
- `tests/` — pytest unit + smoke tests (43 tests, runs in ~5s).

## Get the data

We use the **NFL Big Data Bowl 2023** dataset on Kaggle — it includes the `pff_role` column, which lets us count pass rushers per play and define the blitz label.

1. Create a free Kaggle account at https://www.kaggle.com.
2. Go to https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/data and click **Download All** (you'll need to accept the competition rules).
3. Unzip into `data/`. You should end up with:

```
data/
├── games.csv
├── players.csv
├── plays.csv
├── pffScoutingData.csv
└── week1.csv ... week8.csv      # tracking data (not used by the baseline model)
```

> The baseline model only needs `games.csv`, `plays.csv`, and `pffScoutingData.csv`. The weekly tracking CSVs are large (~1.5 GB each) and only needed if you want to extend the model with player-movement features.

## Train

### With Docker (recommended)
```powershell
cd training
docker build -t blitz-training .
docker run --name blitz-trainer -p 8888:8888 `
  -v "${PWD}:/home/jovyan/work" `
  -v "${PWD}/../data:/home/jovyan/work/../data" `
  blitz-training
docker exec blitz-trainer python train_model.py
```

Open Jupyter at http://localhost:8888 and run `notebooks/01_eda.ipynb` for exploration.

### Without Docker
```powershell
cd training
pip install -r requirements.txt
python train_model.py
```

### Optional: the cost-ratio analysis
```powershell
cd training
python analyze_cost_ratio.py    # downloads nflverse 2021 PBP on first run; needs pyarrow
```
Not part of the training pipeline — `train_model.py` has no external data dependency.
Writes `models/cost_ratio_analysis.json`.

Artifacts land in `training/models/`:
- `blitz_model.joblib`
- `label_encoders.joblib`
- `blitz_calibrator.joblib` — isotonic calibrator, fit on out-of-fold training predictions
- `model_metadata.json` — metrics, decision rule, calibration tables, stability across seeds

The run also prints the cost-ratio sensitivity table and refits the model once per
stability seed, so it takes a few minutes rather than seconds.

## Serve

### Pull from Docker Hub (fastest)
```powershell
docker pull kenjaro/nfl-blitz-predictor:latest   # currently 0.3.0
docker run -p 8000:8000 kenjaro/nfl-blitz-predictor:latest
```

> **0.3.0** keeps 0.2.0's request/response contract but ships a logistic regression in
> place of the random forest, so probabilities shift (usually down) for a given play — see
> [Why logistic regression](#why-logistic-regression).
>
> **0.2.0 was a breaking change from 0.1.0.** The request field `absolute_yardline_number`
> became `yards_to_goal` (and means something different — see the model card), the
> `confidence` and `num_pass_rushers_estimate` response fields were removed, and
> probabilities are now calibrated, so the same play returns a materially lower and more
> honest number than 0.1.0 did. Pin `:0.1.0` if you need the old contract, though its
> probabilities are inflated by roughly 2x.

### Build from source
Copy the artifacts into the serving image:
```powershell
mkdir serving/models
copy training/models/*.joblib serving/models/
copy training/models/*.json serving/models/
```

Build and run:
```powershell
cd serving
docker build -t blitz-api .
docker run -p 8000:8000 blitz-api
```

API docs: http://localhost:8000/docs

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "possession_team": "KC",
    "defensive_team": "BUF",
    "offense_formation": "SHOTGUN",
    "personnel_o": "1 RB, 1 TE, 3 WR",
    "personnel_d": "4 DL, 2 LB, 5 DB",
    "quarter": 3,
    "down": 3,
    "yards_to_go": 8,
    "yards_to_goal": 45,
    "defenders_in_box": 6,
    "pre_snap_home_score": 14,
    "pre_snap_visitor_score": 17,
    "game_clock": "5:32",
    "is_home_offense": 0
  }'
```

Response:
```json
{
  "blitz_probability": 0.1206,
  "will_blitz": false,
  "decision_threshold": 0.25,
  "cost_ratio_fn_to_fp": 3.0,
  "base_rate": 0.246,
  "lift_over_base_rate": 0.49,
  "recommendation": "Blitz probability 12% is below the 25% alert threshold. Standard protection."
}
```

`lift_over_base_rate` is the number to read: 0.49 means this situation is about **half as
blitz-prone as an average pass play**, so the alert correctly stays off. `will_blitz` is an
alert flag at the 0.25 threshold, not a claim about which outcome is more likely — it can
be `true` at a probability well under 0.5, because a missed blitz is assumed 3x costlier
than a false alarm.

For reference, this same request returned **0.53** before calibration (inflated ~2x) and
**0.27** from the random forest. The logistic regression reads this particular spot lower;
see the tail-compression caveat under [Why logistic regression](#why-logistic-regression).

## Tests

43 tests covering feature engineering, label construction, the yardline derivation, API validation, calibration, threshold behavior, coefficient exposure, and football-sanity predictions.

```powershell
# In the training container (has all deps)
MSYS_NO_PATHCONV=1 docker run --rm --user root `
  -v "${PWD}:/home/jovyan/project" -w /home/jovyan/project `
  blitz-training bash -c "pip install --quiet -r tests/requirements.txt fastapi pydantic && python -m pytest tests/ -v"
```

Or locally (Python 3.11+):
```powershell
pip install -r tests/requirements.txt -r serving/requirements.txt
python -m pytest tests/ -v
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health + model status |
| GET | `/model-info` | Metadata: features, metrics, decision rule + cost model, calibration reliability tables, stability across seeds |
| POST | `/predict` | Single prediction |
| POST | `/predict-batch` | Multiple plays in one call |

## Features used

| Group | Features |
|---|---|
| Situational | `quarter`, `down`, `yardsToGo`, `yards_to_goal`, `score_differential`, `game_seconds_remaining`, `half_seconds_remaining`, `is_two_minute`, `is_red_zone`, `is_goal_to_go` |
| Pre-snap box | `defendersInBox` (strongest coefficient: +0.64 per SD, odds ratio 1.90) |
| Teams | `possessionTeam`, `defensiveTeam`, `is_home_offense` |
| Pre-snap look | `offenseFormation`, `personnelO`, `personnelD` |

## Model card (baseline)

- **Algorithm:** `OneHotEncoder(categoricals) + StandardScaler(numerics) -> LogisticRegression(class_weight='balanced', max_iter=1000)`, wrapped in an isotonic calibrator. A RandomForestClassifier (300 trees, max_depth=15, min_samples_leaf=20) is trained every run as the challenger; it ties on AUC and is not shipped.
- **Label definition:** blitz = 5 or more defenders with `pff_role == 'Pass Rush'`
- **Data:** 2021 NFL regular season, weeks 1–8 (8,549 labeled pass plays; 24.6% blitz rate)
- **Train/test split:** `GroupShuffleSplit` by `gameId`, 80/20, random_state=42 — 97 train
  games / 25 test games, no game in both. A stratified play-level split leaks defensive
  scheme between plays in the same game; see [STORY.md](STORY.md).
- **Primary metric:** ROC AUC. Accuracy is reported but is below the majority-class
  floor by design — see [Why not accuracy?](#why-not-accuracy).
- **Feature admission rule:** a field is used only if (1) its value is fixed before the snap
  and (2) it isn't mechanically entangled with the rusher count. `pff_passCoverage` and
  `dropBackType` fail both and are excluded — Cover-0 plays blitz 83.7% of the time because
  the players not covering are the ones rushing. `defendersInBox` passes: 6 in the box still
  blitzes only 21.9%. Reasoning and tables in [STORY.md](STORY.md).
- **Field position:** `yards_to_goal` (1–99, distance to the opponent's goal line), derived
  from `yardlineSide` + `yardlineNumber`. The raw `absoluteYardlineNumber` column is **not**
  usable as a distance — it's a tracking-system field coordinate (11–109) whose orientation
  flips with drive direction. See [STORY.md](STORY.md).
- **Known limitations:**
  - Only ~8 weeks of one season — defensive scheme tendencies can shift year to year
  - No player-tracking features (defender alignment, depth, motion response)
  - Personnel grouping strings are not standardized perfectly across teams
