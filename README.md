# NFL Blitz Predictor

Pre-snap binary classifier: given the offensive context (down, distance, formation, personnel, score, clock), predict whether the defense will blitz (5+ pass rushers).

> **Honest baseline:** Accuracy 0.682, ROC AUC 0.689 on 1,750 plays across 25 held-out games. Trained on 2021 NFL weeks 1–8 from the Big Data Bowl 2023 dataset. See [STORY.md](STORY.md) for the full writeup including the leakage discovery that dropped initial 0.704 → honest 0.689.

Built in the same train-once / serve-many pattern as `titanic-ml-project`:
- `training/` — Jupyter + full ML stack; produces `.joblib` artifacts.
- `serving/` — Slim FastAPI image; loads artifacts and exposes a REST API.
- `tests/` — pytest unit + smoke tests (31 tests, runs in ~5s).

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

Artifacts land in `training/models/`:
- `blitz_model.joblib`
- `label_encoders.joblib`
- `model_metadata.json`

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
    "absolute_yardline_number": 45,
    "pre_snap_home_score": 14,
    "pre_snap_visitor_score": 17,
    "game_clock": "5:32",
    "is_home_offense": 0
  }'
```

Response:
```json
{
  "blitz_probability": 0.68,
  "will_blitz": true,
  "confidence": "moderate",
  "num_pass_rushers_estimate": "likely 5 rushers (blitz)"
}
```

## Tests

31 tests covering feature engineering, label construction, API validation, and football-sanity predictions.

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
| GET | `/model-info` | Metadata (features, metrics, blitz definition) |
| POST | `/predict` | Single prediction |
| POST | `/predict-batch` | Multiple plays in one call |

## Features used

| Group | Features |
|---|---|
| Situational | `quarter`, `down`, `yardsToGo`, `absoluteYardlineNumber`, `score_differential`, `game_seconds_remaining`, `is_two_minute`, `is_red_zone`, `is_goal_to_go` |
| Teams | `possessionTeam`, `defensiveTeam`, `is_home_offense` |
| Pre-snap look | `offenseFormation`, `personnelO`, `personnelD` |

## Model card (baseline)

- **Algorithm:** RandomForestClassifier (300 trees, max_depth=15, class_weight='balanced')
- **Label definition:** blitz = 5 or more defenders with `pff_role == 'Pass Rush'`
- **Data:** 2021 NFL regular season, weeks 1–8 (~8k labeled pass plays)
- **Train/test split:** 80/20 stratified, random_state=42
- **Known limitations:**
  - Only ~8 weeks of one season — defensive scheme tendencies can shift year to year
  - No player-tracking features (defender alignment, depth, motion response)
  - Personnel grouping strings are not standardized perfectly across teams
