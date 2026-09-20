# NFL Blitz Predictor

Pre-snap binary classifier: given the offensive context (down, distance, formation, personnel, score, clock), predict whether the defense will blitz (5+ pass rushers).

> **Honest baseline:** ROC AUC **0.711 ± 0.017** across 10 by-game splits (0.500 = no skill); the shipped model scores 0.689 on its own 25 held-out games. Probabilities are isotonic-calibrated, and the alert threshold is derived from an explicit cost ratio rather than left at 0.5. Blitzes are ~25% of plays, so accuracy is the wrong headline — see [Why not accuracy?](#why-not-accuracy). Trained on 2021 NFL weeks 1–8 from the Big Data Bowl 2023 dataset. Full writeup in [STORY.md](STORY.md).

## Results

Single held-out split (25 games, 1,750 plays, 26.1% blitz rate), reported at the
tuned decision threshold of 0.25:

| Model | ROC AUC | Accuracy | Blitz precision | Blitz recall |
|---|---|---|---|---|
| Always predict "no blitz" | 0.500 | **0.739** | — | 0.00 |
| Logistic regression | 0.682 | 0.655 | — | — |
| **Random forest (shipped)** | **0.689** | 0.619 | 0.37 | 0.66 |

One split of 122 games is a noisy measurement, so the training run repeats the whole
thing over 10 seeds:

| Metric | Mean ± SD over 10 by-game splits |
|---|---|
| Random forest AUC | **0.711 ± 0.017** |
| Logistic regression AUC | 0.708 ± 0.018 |
| Random forest accuracy | 0.700 ± 0.017 |
| Majority-class accuracy | 0.755 ± 0.018 |

The spread matters: a single split can land anywhere from 0.683 to 0.736 on identical
code. Any claim resting on a difference smaller than ~0.02 AUC is not measurable here.

### Why not accuracy?

Because the majority-class model beats us on it — 0.739 vs. 0.619 — and it has no
predictive value whatsoever. Any classifier on a 26% base rate can score 74% by
refusing to ever predict the positive class.

Accuracy drops further once the threshold is tuned, and that's the intended trade: the
model is an **alert**, and a missed blitz costs more than a false alarm. At the tuned
threshold it catches **66% of actual blitzes**, and when it fires it's right **37%** of
the time against a 26.1% base rate. See [The decision rule](#the-decision-rule).

The random forest also beats logistic regression by only **+0.003 ± 0.010 AUC**, winning
6 of 10 seeds. That's a coin flip. Most of the signal here is linear and the ensemble is
not earning its complexity — the honest recommendation would be to ship the logistic
regression and hand a coach the coefficients.

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
| 1:1 | 0.50 | 0.56 | 0.22 | 0.10 |
| 2:1 | 0.33 | 0.48 | 0.44 | 0.24 |
| **3:1 (shipped)** | **0.25** | **0.37** | **0.66** | **0.46** |
| 5:1 | 0.17 | 0.32 | 0.82 | 0.67 |
| 8:1 | 0.11 | 0.28 | 0.95 | 0.89 |

For calibrated probabilities the cost-optimal cutoff has a closed form —
`C_FP / (C_FP + C_FN)`, so 1/(1+3) = **0.25**. A grid search over out-of-fold training
predictions independently picked 0.25, which is the check that the calibration is real.

The threshold is chosen on **out-of-fold training predictions only** (`GroupKFold`
within the training games). Tuning it on the test set would be the same mistake as
tuning a hyperparameter there.

Honest caveat: at 3:1 the tuned threshold and the old default happen to produce the
*same* expected cost on this split (0.5606 per play, from very different confusion
matrices — 157 vs. 214 missed blitzes). So this didn't make the model cheaper. What it
did was make the operating point a stated choice that moves correctly when the cost
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
uncalibrated model predicted a mean blitz probability of **0.434** on a test set whose
actual rate was **0.261** — it was overstating blitz risk by roughly 2×.

Fixed with isotonic regression fit on out-of-fold training predictions:

| | Brier score | Mean predicted | Observed |
|---|---|---|---|
| Raw | 0.2057 | 0.434 | 0.261 |
| **Isotonic (shipped)** | **0.1752** | **0.247** | 0.261 |

14.8% better Brier. Isotonic is monotone, so AUC is unchanged — this fixes what the
numbers *mean*, not how well they rank. Reliability after calibration, from
`/model-info`:

| Predicted bucket | n | Mean predicted | Observed rate |
|---|---|---|---|
| 0.00–0.15 | 511 | 0.100 | 0.137 |
| 0.15–0.25 | 430 | 0.205 | 0.202 |
| 0.25–0.35 | 612 | 0.294 | 0.307 |
| 0.45–0.60 | 153 | 0.517 | 0.542 |
| 0.60–1.00 | 40 | 0.811 | 0.650 |

The API serves calibrated probabilities. `blitz_calibrator.joblib` is a separate
artifact so the two stages stay inspectable.

Built in the same train-once / serve-many pattern as `titanic-ml-project`:
- `training/` — Jupyter + full ML stack; produces `.joblib` artifacts.
- `serving/` — Slim FastAPI image; loads artifacts and exposes a REST API.
- `tests/` — pytest unit + smoke tests (40 tests, runs in ~5s).

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
docker pull kenjaro/nfl-blitz-predictor:latest
docker run -p 8000:8000 kenjaro/nfl-blitz-predictor:latest
```

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
  "blitz_probability": 0.2704,
  "will_blitz": true,
  "decision_threshold": 0.25,
  "cost_ratio_fn_to_fp": 3.0,
  "base_rate": 0.246,
  "lift_over_base_rate": 1.1,
  "recommendation": "Blitz probability 27% is at or above the 25% alert threshold (tuned for a 3:1 cost on missed blitzes). Favor extra protection."
}
```

Note that `will_blitz: true` at a probability of 0.27 is not a contradiction — it's an
alert flag at the 0.25 threshold, not a claim that a blitz is more likely than not. The
`lift_over_base_rate` of 1.1 is the honest read: this situation is barely more
blitz-prone than an average pass play.

(Before calibration this same request returned 0.53, which was inflated by about 2x.)

## Tests

40 tests covering feature engineering, label construction, the yardline derivation, API validation, calibration, threshold behavior, and football-sanity predictions.

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
| Pre-snap box | `defendersInBox` (most important feature, ~19% of total importance) |
| Teams | `possessionTeam`, `defensiveTeam`, `is_home_offense` |
| Pre-snap look | `offenseFormation`, `personnelO`, `personnelD` |

## Model card (baseline)

- **Algorithm:** RandomForestClassifier (300 trees, max_depth=15, min_samples_leaf=20, class_weight='balanced')
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
