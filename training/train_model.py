"""Train the NFL pre-snap blitz classifier on Big Data Bowl 2023 data.

Expects the BDB 2023 CSVs in ../data/:
    games.csv, players.csv, plays.csv, pffScoutingData.csv,
    week1.csv ... week8.csv

The pff_role column in pffScoutingData.csv gives us per-player roles
(Pass Rush, Pass Coverage, Pass Block, etc.) which we use to count
pass rushers per play and define the blitz label.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Blitz = 5 or more pass rushers (standard NFL definition)
BLITZ_THRESHOLD = 5

# --- Decision cost model -----------------------------------------------------
# The consumer is an offensive analyst deciding whether to keep extra protection in.
#   False negative (we say no blitz, defense blitzes): an unblocked rusher gets a
#     free run at the QB. Sack or hurry -- the expensive outcome.
#   False positive (we say blitz, defense rushes four): a back or TE stays in to
#     block instead of running a route. One fewer target, mild EPA cost.
# A missed blitz is worse than a wasted blocker, but not catastrophically so.
#
# 3:1 is still an ASSUMPTION. analyze_cost_ratio.py is the attempt to measure it from
# EPA (nflverse play-by-play joined to these plays) and it does not identify the ratio:
# pff_role == 'Pass Block' is assigned post-snap, so "extra blocker" marks max-protect
# play design (61% play-action) rather than a protection call, and the false-alarm cost
# comes out wrong-signed. What that analysis does establish: a sack costs ~2.06 EPA, a
# blitz adds 2.7 points of sack risk, but a blitz's NET effect on offensive EPA is only
# about -0.036 (95% CI -0.094 to +0.025) because it also gives up coverage.
#
# So: the ratio is an input to the decision, not a finding. The training run prints a
# sensitivity table across ratios, and both the ratio and its provenance go into the
# metadata. Do not quote 3:1 as if it were measured.
COST_FALSE_NEGATIVE = 3.0
COST_FALSE_POSITIVE = 1.0

# For *calibrated* probabilities the cost-optimal cutoff has a closed form:
# alert whenever p * C_FN > (1 - p) * C_FP, i.e. p > C_FP / (C_FP + C_FN).
# This is only valid if the probabilities mean what they say, which is why the
# isotonic calibration step below is a prerequisite and not a nice-to-have.
DECISION_THRESHOLD = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)

# Number of by-game splits used to put error bars on the headline metrics.
N_STABILITY_SEEDS = 10
CALIBRATION_FOLDS = 5

CATEGORICAL_FEATURES = [
    "possessionTeam",
    "defensiveTeam",
    "offenseFormation",
    "personnelO",
    "personnelD",
]

NUMERIC_FEATURES = [
    "quarter",
    "down",
    "yardsToGo",
    "yards_to_goal",
    "defendersInBox",
    "preSnapHomeScore",
    "preSnapVisitorScore",
    "score_differential",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "is_home_offense",
    "is_two_minute",
    "is_red_zone",
    "is_goal_to_go",
]


def gameclock_to_seconds(clock: str) -> int:
    """'14:32' -> 872 (seconds remaining in the quarter)."""
    if pd.isna(clock):
        return 0
    parts = str(clock).split(":")
    if len(parts) != 2:
        return 0
    return int(parts[0]) * 60 + int(parts[1])


def build_blitz_labels(pff: pd.DataFrame) -> pd.DataFrame:
    """Count pass rushers per play and label blitz (>=5 rushers)."""
    rushers = (
        pff[pff["pff_role"] == "Pass Rush"]
        .groupby(["gameId", "playId"])
        .size()
        .rename("num_pass_rushers")
        .reset_index()
    )
    rushers["was_blitz"] = (rushers["num_pass_rushers"] >= BLITZ_THRESHOLD).astype(int)
    return rushers


def yards_to_goal(df: pd.DataFrame) -> pd.Series:
    """Yards from the ball to the *opponent's* goal line, 1-99.

    Do not use absoluteYardlineNumber for this. That column is a raw tracking-system
    field coordinate (range 11-109) whose orientation flips with the direction the
    offense happens to be driving -- every play in the BDB 2023 data satisfies either
    abs == yards_to_goal + 10 or abs == 110 - yards_to_goal, roughly half each. So the
    same field position encodes as two different numbers depending on the possessing
    team's end of the field, which scrambles any distance-to-goal signal.

    yardlineSide + yardlineNumber are direction-independent, so derive from those:
    yardlineSide is the team whose half the ball is on, and it is null exactly at the
    50 (where both branches give 50 anyway).
    """
    return pd.Series(
        np.where(
            df["yardlineSide"] == df["possessionTeam"],
            100 - df["yardlineNumber"],  # own half: cross midfield first
            df["yardlineNumber"],        # opponent half: number *is* the distance
        ),
        index=df.index,
    )


def engineer_features(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Add derived pre-snap features."""
    df = plays.merge(games[["gameId", "homeTeamAbbr", "visitorTeamAbbr"]], on="gameId", how="left")

    df["is_home_offense"] = (df["possessionTeam"] == df["homeTeamAbbr"]).astype(int)

    df["yards_to_goal"] = yards_to_goal(df)

    df["score_differential"] = np.where(
        df["is_home_offense"] == 1,
        df["preSnapHomeScore"] - df["preSnapVisitorScore"],
        df["preSnapVisitorScore"] - df["preSnapHomeScore"],
    )

    df["clock_seconds"] = df["gameClock"].apply(gameclock_to_seconds)
    df["half_seconds_remaining"] = np.where(
        df["quarter"].isin([1, 3]),
        df["clock_seconds"] + 900,
        df["clock_seconds"],
    )
    df["game_seconds_remaining"] = np.where(
        df["quarter"] <= 2,
        df["half_seconds_remaining"] + 1800,
        df["half_seconds_remaining"],
    )

    df["is_two_minute"] = ((df["quarter"].isin([2, 4])) & (df["clock_seconds"] <= 120)).astype(int)
    df["is_red_zone"] = (df["yards_to_goal"] <= 20).astype(int)
    df["is_goal_to_go"] = (df["yardsToGo"] >= df["yards_to_goal"]).astype(int)

    return df


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def make_logistic_regression() -> Pipeline:
    """One-hot the categoricals: label-encoding imposes a fake ordinal
    relationship that would unfairly hurt a linear model."""
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def out_of_fold_probabilities(X, y, groups, n_splits=CALIBRATION_FOLDS):
    """Probabilities for every training row, each predicted by a model that never
    saw that row's game. These are what the calibrator and the threshold check are
    fit on -- using the held-out test set for either would be tuning on test.
    """
    oof = np.zeros(len(y))
    for fit_idx, pred_idx in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        fold_model = make_random_forest().fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof[pred_idx] = fold_model.predict_proba(X.iloc[pred_idx])[:, 1]
    return oof


def expected_cost(y_true, proba, threshold, c_fn=COST_FALSE_NEGATIVE,
                  c_fp=COST_FALSE_POSITIVE) -> float:
    """Average cost per play at a given alert threshold."""
    pred = (proba >= threshold).astype(int)
    false_neg = int(((pred == 0) & (y_true == 1)).sum())
    false_pos = int(((pred == 1) & (y_true == 0)).sum())
    return (c_fn * false_neg + c_fp * false_pos) / len(y_true)


def reliability_table(y_true, proba, edges=(0.0, 0.15, 0.25, 0.35, 0.45, 0.60, 1.01)):
    """Predicted vs. observed blitz rate per probability bucket."""
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{lo:.2f}-{min(hi, 1.0):.2f}",
            "n": int(mask.sum()),
            "mean_predicted": round(float(proba[mask].mean()), 4),
            "observed_rate": round(float(np.asarray(y_true)[mask].mean()), 4),
        })
    return rows


def stability_across_splits(X, y, groups, n_seeds=N_STABILITY_SEEDS):
    """Re-run the whole by-game split n_seeds times.

    A single 80/20 split of 122 games leaves ~25 games in test, and which 25 you
    draw moves the metrics a lot. One split cannot tell a real effect from the luck
    of the draw, so report mean +/- std instead of a single number. This also
    re-measures the leaky play-level split each time, so the leakage claim gets
    error bars rather than resting on one comparison.
    """
    rf_auc, lr_auc, rf_acc, majority_acc, leaky_auc = [], [], [], [], []
    for seed in range(n_seeds):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(X, y, groups=groups))
        rf = make_random_forest().fit(X.iloc[tr], y.iloc[tr])
        rf_auc.append(roc_auc_score(y.iloc[te], rf.predict_proba(X.iloc[te])[:, 1]))
        rf_acc.append(accuracy_score(y.iloc[te], rf.predict(X.iloc[te])))
        majority_acc.append(1.0 - float(y.iloc[te].mean()))

        lr = make_logistic_regression().fit(X.iloc[tr], y.iloc[tr])
        lr_auc.append(roc_auc_score(y.iloc[te], lr.predict_proba(X.iloc[te])[:, 1]))

        # Same model, same seed, but the textbook play-level split -- the one that
        # lets plays from a single game land in both train and test.
        Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed)
        leaky = make_random_forest().fit(Xl_tr, yl_tr)
        leaky_auc.append(roc_auc_score(yl_te, leaky.predict_proba(Xl_te)[:, 1]))

        print(f"    seed {seed}: by-game AUC {rf_auc[-1]:.4f}   play-level AUC {leaky_auc[-1]:.4f}")

    def summarize(values):
        a = np.array(values)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                "min": float(a.min()), "max": float(a.max())}

    leak_delta = np.array(leaky_auc) - np.array(rf_auc)
    rf_vs_lr = np.array(rf_auc) - np.array(lr_auc)
    return {
        "n_seeds": n_seeds,
        "random_forest_auc": summarize(rf_auc),
        "logistic_regression_auc": summarize(lr_auc),
        "random_forest_accuracy": summarize(rf_acc),
        "majority_class_accuracy": summarize(majority_acc),
        "leaky_play_level_auc": summarize(leaky_auc),
        "leakage_delta": {
            **summarize(leak_delta),
            "std_error": float(leak_delta.std(ddof=1) / np.sqrt(n_seeds)),
            "seeds_where_leaky_higher": int((leak_delta > 0).sum()),
        },
        "random_forest_minus_logistic": {
            **summarize(rf_vs_lr),
            "seeds_where_rf_higher": int((rf_vs_lr > 0).sum()),
        },
    }


def main():
    print("Loading BDB 2023 data...")
    games = pd.read_csv(DATA_DIR / "games.csv")
    plays = pd.read_csv(DATA_DIR / "plays.csv")
    pff = pd.read_csv(DATA_DIR / "pffScoutingData.csv")

    print(f"  games:  {len(games):>7,} rows")
    print(f"  plays:  {len(plays):>7,} rows")
    print(f"  pff:    {len(pff):>7,} rows")

    print("\nBuilding blitz labels from pff_role...")
    labels = build_blitz_labels(pff)
    print(f"  labeled plays: {len(labels):>7,}")
    print(f"  blitz rate:    {labels['was_blitz'].mean():.1%}")

    print("\nMerging features and labels...")
    df = engineer_features(plays, games).merge(labels, on=["gameId", "playId"], how="inner")
    df = df.dropna(subset=CATEGORICAL_FEATURES + ["was_blitz"])
    print(f"  final dataset: {len(df):>7,} rows")

    print("\nEncoding categorical features...")
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df["was_blitz"]
    groups = df["gameId"]

    # Split by game so no game appears in both train and test.
    # Random play-level splits leak defensive scheme between halves.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"  train games: {groups.iloc[train_idx].nunique()}")
    print(f"  test games:  {groups.iloc[test_idx].nunique()}")
    print(f"  train plays: {len(X_train):,}  blitz rate: {y_train.mean():.1%}")
    print(f"  test plays:  {len(X_test):,}  blitz rate: {y_test.mean():.1%}")

    print("\nFitting calibrator + threshold check on training folds only...")
    oof_proba = out_of_fold_probabilities(X_train, y_train, groups.iloc[train_idx])
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_proba, y_train)
    oof_calibrated = calibrator.predict(oof_proba)
    print(f"  out-of-fold AUC: {roc_auc_score(y_train, oof_proba):.4f}")

    # The analytic threshold assumes calibrated probabilities. Confirm it against a
    # brute-force sweep on the out-of-fold predictions; if they disagree by much,
    # the calibration is not doing its job and the closed form isn't trustworthy.
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    grid_threshold = float(grid[int(np.argmin(
        [expected_cost(y_train.values, oof_calibrated, t) for t in grid]))])
    print(f"  cost ratio FN:FP = {COST_FALSE_NEGATIVE:.0f}:{COST_FALSE_POSITIVE:.0f}")
    print(f"  analytic threshold C_FP/(C_FP+C_FN) = {DECISION_THRESHOLD:.3f}")
    print(f"  grid-searched on out-of-fold        = {grid_threshold:.3f}")
    if abs(grid_threshold - DECISION_THRESHOLD) > 0.05:
        print("  WARNING: analytic and empirical thresholds disagree -- check calibration")

    print("\nTraining Random Forest...")
    model = make_random_forest()
    model.fit(X_train, y_train)

    print("\nEvaluating on held-out games...")
    y_proba_raw = model.predict_proba(X_test)[:, 1]
    y_proba = calibrator.predict(y_proba_raw)   # what the API serves
    # Isotonic regression is monotone, so it cannot change the ranking and AUC is
    # identical up to ties. It only fixes what the numbers *mean*.
    auc = roc_auc_score(y_test, y_proba_raw)

    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)   # shipped decision rule
    y_pred_default = (y_proba_raw >= 0.5).astype(int)       # sklearn's default, for contrast

    accuracy = accuracy_score(y_test, y_pred)

    # Blitz is the minority class (~26% of plays), so accuracy is dominated by
    # the no-blitz majority and a "never blitz" model scores higher than we do.
    # Rank-based (AUC) and blitz-class (precision/recall) metrics are the ones
    # that actually say whether the model found signal. Keep all of them.
    test_blitz_rate = float(y_test.mean())
    majority_accuracy = 1.0 - test_blitz_rate
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[1], zero_division=0
    )
    blitz_precision, blitz_recall, blitz_f1 = (
        float(precision[0]), float(recall[0]), float(f1[0])
    )

    brier_raw = float(brier_score_loss(y_test, y_proba_raw))
    brier_calibrated = float(brier_score_loss(y_test, y_proba))

    print(f"  ROC AUC:  {auc:.4f}   (0.500 = no skill)")
    print(f"  accuracy: {accuracy:.4f}   (majority-class floor {majority_accuracy:.4f})")
    print(f"  at the tuned threshold {DECISION_THRESHOLD:.2f}:")
    print(f"    blitz precision/recall/F1: {blitz_precision:.3f} / {blitz_recall:.3f} / {blitz_f1:.3f}")
    print(f"    precision lift over base rate: {blitz_precision / test_blitz_rate:.2f}x")
    print(f"    alert rate: {y_pred.mean():.3f}  (base rate {test_blitz_rate:.3f})")

    print("\n  Calibration:")
    print(f"    Brier  raw {brier_raw:.4f} -> calibrated {brier_calibrated:.4f}"
          f"  ({100 * (1 - brier_calibrated / brier_raw):.1f}% better)")
    print(f"    mean predicted  raw {y_proba_raw.mean():.3f} -> calibrated "
          f"{y_proba.mean():.3f}   (observed {test_blitz_rate:.3f})")
    print(f"    {'bucket':>12} {'n':>6} {'predicted':>10} {'observed':>9}")
    for row in reliability_table(y_test.values, y_proba):
        print(f"    {row['bin']:>12} {row['n']:>6} {row['mean_predicted']:>10.3f} "
              f"{row['observed_rate']:>9.3f}")

    print("\n  Cost per play (FN:FP = "
          f"{COST_FALSE_NEGATIVE:.0f}:{COST_FALSE_POSITIVE:.0f}), lower is better:")
    decision_costs = {
        "always_no_blitz": expected_cost(y_test.values, np.zeros(len(y_test)), 0.5),
        "always_blitz": expected_cost(y_test.values, np.ones(len(y_test)), 0.5),
        "raw_probability_at_0.50": expected_cost(y_test.values, y_proba_raw, 0.5),
        "calibrated_at_tuned_threshold": expected_cost(y_test.values, y_proba, DECISION_THRESHOLD),
    }
    for name, c in decision_costs.items():
        print(f"    {name:<32} {c:.4f}")

    print("\n  Cost-ratio sensitivity (threshold from the closed form, metrics on test):")
    print(f"    {'FN:FP':>7} {'thr':>6} {'precision':>10} {'recall':>8} {'alert rate':>11}")
    sensitivity = []
    for ratio in (1, 2, 3, 5, 8):
        thr = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + ratio)
        pred = (y_proba >= thr).astype(int)
        p, r, _, _ = precision_recall_fscore_support(y_test, pred, labels=[1], zero_division=0)
        sensitivity.append({"cost_ratio_fn_to_fp": ratio, "threshold": round(thr, 4),
                            "blitz_precision": round(float(p[0]), 4),
                            "blitz_recall": round(float(r[0]), 4),
                            "alert_rate": round(float(pred.mean()), 4)})
        print(f"    {ratio:>5}:1 {thr:>6.2f} {p[0]:>10.3f} {r[0]:>8.3f} {pred.mean():>11.3f}")

    print("\n  Classification report at the tuned threshold:")
    print(classification_report(y_test, y_pred, target_names=["no_blitz", "blitz"]))
    print("Confusion matrix (tuned threshold):")
    print(confusion_matrix(y_test, y_pred))
    print("Confusion matrix (default 0.50 on raw probabilities, for contrast):")
    print(confusion_matrix(y_test, y_pred_default))

    importances = sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop 10 features by importance:")
    for name, imp in importances[:10]:
        print(f"  {name:<30} {imp:.4f}")

    print("\nBaseline comparison (same train/test split, by-game)...")
    baseline_metrics = {}

    # Majority-class baseline: is the model actually better than "always predict no-blitz"?
    majority = DummyClassifier(strategy="most_frequent", random_state=42)
    majority.fit(X_train, y_train)
    maj_pred = majority.predict(X_test)
    maj_proba = majority.predict_proba(X_test)[:, 1]
    baseline_metrics["majority_class"] = {
        "accuracy": float(accuracy_score(y_test, maj_pred)),
        "roc_auc": float(roc_auc_score(y_test, maj_proba)),
    }

    logreg_pipeline = make_logistic_regression()
    logreg_pipeline.fit(X_train, y_train)
    lr_pred = logreg_pipeline.predict(X_test)
    lr_proba = logreg_pipeline.predict_proba(X_test)[:, 1]
    baseline_metrics["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_test, lr_pred)),
        "roc_auc": float(roc_auc_score(y_test, lr_proba)),
    }

    baseline_metrics["random_forest"] = {
        "accuracy": float(accuracy),
        "roc_auc": float(auc),
    }

    print(f"\n  test-set blitz rate: {y_test.mean():.1%}  (majority-class accuracy floor)")
    print(f"\n  {'Model':<22}{'Accuracy':>10}{'ROC AUC':>10}")
    print(f"  {'-' * 42}")
    for name, m in baseline_metrics.items():
        label = name.replace("_", " ").title()
        print(f"  {label:<22}{m['accuracy']:>10.4f}{m['roc_auc']:>10.4f}")

    print(f"\nStability across {N_STABILITY_SEEDS} by-game splits "
          "(this refits the model once per seed)...")
    stability = stability_across_splits(X, y, groups)
    print()
    for key in ("random_forest_auc", "logistic_regression_auc",
                "random_forest_accuracy", "majority_class_accuracy",
                "leaky_play_level_auc"):
        s = stability[key]
        print(f"  {key:<28} {s['mean']:.4f} +/- {s['std']:.4f}   "
              f"[{s['min']:.4f}, {s['max']:.4f}]")
    ld = stability["leakage_delta"]
    print(f"\n  leakage (play-level AUC - by-game AUC): {ld['mean']:+.4f} "
          f"+/- {ld['std_error']:.4f} (SE)")
    print(f"    play-level split scored higher in {ld['seeds_where_leaky_higher']}"
          f"/{stability['n_seeds']} seeds")
    rl = stability["random_forest_minus_logistic"]
    print(f"  random forest - logistic regression AUC: {rl['mean']:+.4f} "
          f"+/- {rl['std']:.4f}")
    print(f"    random forest won {rl['seeds_where_rf_higher']}/{stability['n_seeds']} seeds")

    print("\nSaving artifacts...")
    joblib.dump(model, MODEL_DIR / "blitz_model.joblib")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.joblib")
    joblib.dump(calibrator, MODEL_DIR / "blitz_calibrator.joblib")

    metadata = {
        "model_type": "RandomForestClassifier",
        "target": "was_blitz",
        "blitz_definition": f">={BLITZ_THRESHOLD} pass rushers (pff_role == 'Pass Rush')",
        "feature_order": feature_cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "metrics": {
            "roc_auc": float(auc),
            "accuracy": float(accuracy),
            "majority_class_accuracy": float(majority_accuracy),
            "blitz_precision": blitz_precision,
            "blitz_recall": blitz_recall,
            "blitz_f1": blitz_f1,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "blitz_rate": float(y.mean()),
            "test_blitz_rate": test_blitz_rate,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "confusion_matrix_at_default_0.50": confusion_matrix(y_test, y_pred_default).tolist(),
            "brier_raw": brier_raw,
            "brier_calibrated": brier_calibrated,
            "primary_metric": "roc_auc",
            "accuracy_note": (
                "Accuracy is below the majority-class floor by design: the decision "
                "threshold is tuned to a 3:1 cost on missed blitzes, which buys recall "
                "with accuracy. Use roc_auc, the blitz-class precision/recall, and "
                "expected cost per play instead."
            ),
        },
        "decision_rule": {
            "threshold": DECISION_THRESHOLD,
            "cost_false_negative": COST_FALSE_NEGATIVE,
            "cost_false_positive": COST_FALSE_POSITIVE,
            "threshold_source": (
                "Closed form C_FP/(C_FP+C_FN) on isotonic-calibrated probabilities, "
                "cross-checked against a grid search on out-of-fold training "
                f"predictions (grid picked {grid_threshold:.2f})."
            ),
            "grid_threshold_on_out_of_fold": grid_threshold,
            "rationale": (
                "Consumer is an offensive analyst deciding whether to keep extra "
                "protection in. A missed blitz gives up a free rusher; a false alarm "
                "costs one route runner. 3:1 is an ASSUMPTION, not a measurement -- "
                "see cost_ratio_sensitivity for how the operating point moves."
            ),
            "cost_ratio_provenance": (
                "Asserted, not measured. analyze_cost_ratio.py attempts to derive it "
                "from EPA and fails to identify it (pff_role is post-snap, so 'extra "
                "blocker' marks max-protect play design rather than a protection call; "
                "the false-alarm cost comes out wrong-signed). That analysis does show "
                "a sack costs ~2.06 EPA and a blitz adds 2.7pp of sack risk, but the "
                "net EPA effect of a blitz is only ~-0.036 (95% CI -0.094 to +0.025), "
                "which bounds what any blitz warning can be worth. See "
                "models/cost_ratio_analysis.json."
            ),
            "cost_ratio_sensitivity": sensitivity,
            "expected_cost_per_play": {k: round(v, 4) for k, v in decision_costs.items()},
        },
        "calibration": {
            "method": "IsotonicRegression fit on out-of-fold training predictions",
            "artifact": "blitz_calibrator.joblib",
            "brier_raw": brier_raw,
            "brier_calibrated": brier_calibrated,
            "note": (
                "class_weight='balanced' inflates raw probabilities -- the uncalibrated "
                "model predicted a mean of "
                f"{y_proba_raw.mean():.3f} against an observed rate of {test_blitz_rate:.3f}. "
                "Serve calibrated probabilities; isotonic is monotone so ranking (AUC) "
                "is unchanged."
            ),
            "reliability_calibrated": reliability_table(y_test.values, y_proba),
            "reliability_raw": reliability_table(y_test.values, y_proba_raw),
        },
        "baseline_comparison": baseline_metrics,
        "stability": stability,
        "data_source": "NFL Big Data Bowl 2023 (Kaggle)",
        "training_seasons": "2021 weeks 1-8",
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Artifacts in {MODEL_DIR}/")


if __name__ == "__main__":
    main()
