# How I Built the NFL Blitz Predictor

A writeup of the decisions, dead ends, and one important fix that made the metrics honest.

## The problem

Given pre-snap offensive context (down, distance, formation, personnel, score, clock), predict whether the defense is going to blitz (5+ pass rushers). This is the kind of model an offensive coordinator's analyst would want at the line of scrimmage — but it's also genuinely hard, because *defenses try to disguise blitzes*. That's a real ceiling: if a model is too good, something is leaking.

## Why this dataset (and not the obvious one)

The obvious first instinct was `nfl_data_py` (nflverse play-by-play). It has 20+ years of plays, easy install, well documented. But it has no per-play blitz label. The closest proxies are `qb_hit` or `sack` — those measure outcomes ("the QB got pressured"), not the defensive *call*. A model trained on those would really be predicting "did the OL break down," which is a different problem.

The NFL Big Data Bowl 2023 dataset (Kaggle) was a better fit. It includes a `pff_role` column with per-player labels — `Pass Rush`, `Pass Coverage`, `Pass Block`, etc. That gives a clean per-play count of rushers, and the standard NFL definition `rushers >= 5` becomes the binary label.

The tradeoff: only 2021 weeks 1–8 (~8.5k labeled pass plays vs. ~1M plays in nflverse). I made the call that label quality > sample size for a first model. If I were extending this, I'd combine: use BDB to train, then validate on a separate season's PBP with a heuristic label, and see how the model degrades.

## Features

All pre-snap, all from `plays.csv` (the offense's view of the world before the ball is snapped):

| Group | Features |
|---|---|
| Situation | down, yardsToGo, absoluteYardlineNumber, quarter, gameClock, score |
| Engineered | score_differential, game_seconds_remaining, is_two_minute, is_red_zone, is_goal_to_go |
| Teams | possessionTeam, defensiveTeam, is_home_offense |
| Pre-snap look | offenseFormation, personnelO, personnelD, **defendersInBox** |

`defendersInBox` ended up being the single most important feature (~19% of total importance) — which is football-obvious in hindsight: when the offense lines up and there are 7 guys near the line of scrimmage, the defense has *fewer* coverage players left, so a blitz is more likely.

I deliberately excluded `pff_passCoverage` and `dropBackType` even though they're in `plays.csv`. Those are technically pre-snap states, but PFF tags them after watching the play — using them would be a soft form of label leakage at inference time (the offense doesn't *know* the coverage call pre-snap).

## The leakage I almost shipped

First run used `train_test_split(stratify=y, random_state=42)` — the textbook default. Got accuracy **0.701, AUC 0.704**. Felt reasonable.

Then I realized: a random play-level split puts plays from the same game in both train and test. And defensive scheme is consistent within a game — the same DC is calling the same plays from the same playbook. So the model was learning a team's *Sunday-specific tendency* and being tested on more plays from that same Sunday.

Fixed it with `GroupShuffleSplit(test_size=0.2, ...)` grouping by `gameId`. That guarantees no game appears in both sets: 97 train games / 25 test games, no overlap. The honest numbers:

| Metric | Random split (leaky) | By-game split (honest) | Δ |
|---|---|---|---|
| Accuracy | 0.701 | **0.682** | −1.9 pts |
| ROC AUC | 0.704 | **0.689** | −0.015 |

Less than 2 points of leakage, which feels small — but if I'd shipped the first number on a resume I'd be quoting a metric that doesn't generalize to unseen games. That's the kind of thing that fails an interview question.

## Sanity check: are the predictions football-sensible?

Three test scenarios after deploying the API:

| Situation | Predicted blitz prob | Reads |
|---|---|---|
| 3rd & 8, mid-field, 6-man box | 51.6% | Borderline — model is appropriately uncertain |
| 4th & 1 goal-to-go, 9-man box, late game | 73.1% | Obvious blitz situation — model catches it |
| 1st & 10 from own 25, 5-man box, game start | 34.9% | Vanilla coverage situation — correctly leans no-blitz |

Calibration isn't great (the model is more confident than it should be on borderline cases), but the *direction* is right across the spectrum. For a portfolio model with limited data, that's the bar.

## Limitations I'd call out in an interview

1. **One season, 8 weeks.** Defensive coordinators change scheme between seasons. The model would degrade fast on 2022+ data without retraining.
2. **No player-tracking features.** The `week*.csv` files contain 10Hz tracking for every player on every play — that's where the next big AUC gains live (defender alignment depth, motion response, walk-up timing). Skipped for scope; it would roughly triple the project size.
3. **Class imbalance handled with `class_weight='balanced'`**, not threshold tuning. A real production model would tune the decision threshold to the cost of false positives vs. false negatives in the actual use case (e.g. an offensive analyst probably prefers high recall on blitz, even at lower precision).
4. **Personnel strings aren't standardized.** "1 RB, 1 TE, 3 WR" is treated as a categorical token. A better approach would parse it into integer columns (n_rb, n_te, n_wr, n_db).

## What I'd do next

In order of expected payoff:

1. **Add team historical blitz rate as a feature** — compute defensive team's training-set blitz rate, join in. Captures coaching tendency without leakage.
2. **Parse personnel into integer columns** — frees the model from learning that "1 RB, 1 TE, 3 WR" and "1RB 1TE 3WR" are the same.
3. **Try LightGBM / XGBoost** — usually +1–3 AUC points on tabular features like these.
4. **Add player-tracking features** — defender alignment depth, motion response. This is the real ceiling-mover but is its own project.
5. **Threshold tuning + calibration plot** — pick an operating point that matches a real use case, and show that the predicted probabilities actually calibrate (with isotonic regression if not).

## Stack

- **Data:** NFL Big Data Bowl 2023 (Kaggle)
- **Modeling:** scikit-learn (RandomForestClassifier, GroupShuffleSplit, LabelEncoder)
- **Serving:** FastAPI + uvicorn on Python 3.11
- **Containerization:** Docker — separate training and serving images, train produces `.joblib` artifacts baked into the serving image
- **Distribution:** `kenjaro/nfl-blitz-predictor` on Docker Hub
