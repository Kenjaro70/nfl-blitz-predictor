"""Try to ground the 3:1 false-negative:false-positive cost ratio in EPA.

The decision threshold in train_model.py comes from a cost ratio: how much worse is a
missed blitz than a false alarm? That ratio was asserted (3:1). This script is the
attempt to measure it instead, and a record of why the measurement does not identify it.

Approach: join BDB 2023 plays to nflverse play-by-play (which carries EPA) and compare
the four cells of {defense blitzed or not} x {offense kept extra protection or not}.

    cost of a missed blitz = EPA(blitz, heavy protection) - EPA(blitz, light)
    cost of a false alarm  = EPA(no blitz, light)         - EPA(no blitz, heavy)

RESULT: this does not work, and the failure is instructive rather than fixable by
tuning. `pff_role == 'Pass Block'` is assigned from what a player *did* after the snap,
so "6+ blockers" is not a pre-snap protection call -- it tracks play design. The
heavy-protection cells are 61% play-action with 12.9 mean air yards against 12% and 9.0
for light protection. They are max-protect deep shots, and they post *higher* EPA. The
false-alarm cost comes out wrong-signed as a result.

What the script does establish, cleanly:
  * a sack costs about -2.06 EPA relative to a non-sack pass play
  * a blitz raises the sack rate from 6.1% to 8.8%
  * but a blitz's *net* effect on offensive EPA is only about -0.036, 95% CI
    [-0.100, +0.024] -- the sack upside is largely offset by the coverage the blitz
    gives up, which bounds how much any blitz-warning system can be worth

Requires the nflverse 2021 play-by-play parquet (downloaded on first run) and pyarrow.
Not part of the training pipeline -- train_model.py has no external data dependency.

Usage:  python analyze_cost_ratio.py
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "models"
PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
           "play_by_play_2021.parquet")
PBP_CACHE = Path(__file__).parent / "pbp_2021.parquet"

BLITZ_THRESHOLD = 5      # 5+ pass rushers
HEAVY_PROTECTION = 6     # 6+ pass blockers, i.e. someone beyond the five OL blocked
N_BOOTSTRAP = 2000


def load_play_by_play() -> pd.DataFrame:
    if not PBP_CACHE.exists():
        print(f"Downloading nflverse 2021 play-by-play -> {PBP_CACHE.name} ...")
        urllib.request.urlretrieve(PBP_URL, PBP_CACHE)
    pbp = pd.read_parquet(PBP_CACHE, columns=[
        "old_game_id", "play_id", "epa", "play_type", "sack", "qb_hit",
        "pass_length", "air_yards", "complete_pass"])
    pbp["old_game_id"] = pd.to_numeric(pbp["old_game_id"], errors="coerce")
    return pbp


def build_joined_frame() -> pd.DataFrame:
    plays = pd.read_csv(DATA_DIR / "plays.csv")
    pff = pd.read_csv(DATA_DIR / "pffScoutingData.csv")

    # Per-play counts of who rushed, who blocked, who ran a route.
    roles = (pff.pivot_table(index=["gameId", "playId"], columns="pff_role",
                             values="nflId", aggfunc="count")
             .fillna(0)
             .rename(columns={"Pass Rush": "n_rush", "Pass Block": "n_block",
                              "Pass Route": "n_route"})
             .reset_index())

    df = plays.merge(roles, on=["gameId", "playId"], how="inner").merge(
        load_play_by_play(), left_on=["gameId", "playId"],
        right_on=["old_game_id", "play_id"], how="left")

    matched = df["epa"].notna().mean()
    print(f"EPA join rate: {df['epa'].notna().sum()}/{len(df)} = {matched:.1%}")

    df = df[(df["play_type"] == "pass") & df["epa"].notna()].copy()
    df["blitz"] = (df["n_rush"] >= BLITZ_THRESHOLD).astype(int)
    df["heavy"] = (df["n_block"] >= HEAVY_PROTECTION).astype(int)
    return df


def cell_contrasts(df: pd.DataFrame):
    """(cost of a missed blitz, cost of a false alarm) in EPA."""
    mean_epa = df.groupby(["blitz", "heavy"])["epa"].mean()
    cost_fn = mean_epa[(1, 1)] - mean_epa[(1, 0)]
    cost_fp = mean_epa[(0, 0)] - mean_epa[(0, 1)]
    return cost_fn, cost_fp


def cluster_bootstrap(df: pd.DataFrame, statistic, n=N_BOOTSTRAP, seed=42):
    """Resample whole games, not plays -- plays within a game are not independent."""
    rng = np.random.default_rng(seed)
    games = df["gameId"].unique()
    out = []
    for _ in range(n):
        sample = df[df["gameId"].isin(rng.choice(games, len(games), replace=True))]
        try:
            out.append(statistic(sample))
        except KeyError:
            continue
    return np.array(out)


def main():
    df = build_joined_frame()
    print(f"pass plays with EPA: {len(df):,}\n")

    print("=" * 78)
    print("The 2x2: defense's rush vs. offense's protection")
    print("=" * 78)
    print(f"  {'cell':<22}{'n':>6}{'EPA':>8}{'sack%':>7}{'routes':>8}"
          f"{'air yds':>9}{'play-act%':>10}")
    for (blitz, heavy), g in df.groupby(["blitz", "heavy"]):
        name = f"{'blitz' if blitz else '4-man'} / {'heavy' if heavy else 'light'}"
        print(f"  {name:<22}{len(g):>6}{g['epa'].mean():>8.3f}"
              f"{g['sack'].mean() * 100:>6.1f}%{g['n_route'].mean():>8.2f}"
              f"{g['air_yards'].mean():>9.2f}{g['pff_playAction'].mean() * 100:>9.0f}%")

    cost_fn, cost_fp = cell_contrasts(df)
    print(f"\n  cost of a missed blitz = {cost_fn:+.4f} EPA")
    print(f"  cost of a false alarm  = {cost_fp:+.4f} EPA")
    print(f"  implied ratio          = {cost_fn / cost_fp:.2f}:1")

    boot_fn = cluster_bootstrap(df, lambda s: cell_contrasts(s)[0])
    boot_fp = cluster_bootstrap(df, lambda s: cell_contrasts(s)[1])
    print(f"\n  cluster-bootstrapped by game ({N_BOOTSTRAP} reps):")
    print(f"    cost_FN  95% CI [{np.percentile(boot_fn, 2.5):+.4f}, "
          f"{np.percentile(boot_fn, 97.5):+.4f}]  (spans zero)")
    print(f"    cost_FP  95% CI [{np.percentile(boot_fp, 2.5):+.4f}, "
          f"{np.percentile(boot_fp, 97.5):+.4f}]  (reliably NEGATIVE -- wrong sign)")

    print("\n  NOT IDENTIFIED. 'Heavy protection' is not a protection call: the")
    print("  heavy cells run ~61% play-action and ~13 air yards against ~12% and ~9")
    print("  for light. These are max-protect deep shots, so the extra blocker is a")
    print("  marker of aggressive play design, and pff_role is assigned post-snap.")
    print("  The offense's pre-snap protection *intent* is unobserved here.")

    print("\n" + "=" * 78)
    print("What is cleanly measurable")
    print("=" * 78)
    sacks, no_sacks = df[df["sack"] == 1], df[df["sack"] == 0]
    sack_cost = sacks["epa"].mean() - no_sacks["epa"].mean()
    print(f"  EPA | sack taken            {sacks['epa'].mean():+.3f}  (n={len(sacks)})")
    print(f"  EPA | no sack               {no_sacks['epa'].mean():+.3f}  (n={len(no_sacks)})")
    print(f"  => EPA cost of a sack       {sack_cost:+.3f}")

    blitzed, not_blitzed = df[df["blitz"] == 1], df[df["blitz"] == 0]
    sack_rate_delta = blitzed["sack"].mean() - not_blitzed["sack"].mean()
    print(f"\n  sack rate | blitz           {blitzed['sack'].mean() * 100:.1f}%")
    print(f"  sack rate | 4-man rush      {not_blitzed['sack'].mean() * 100:.1f}%")
    print(f"  => blitz adds              {sack_rate_delta * 100:+.1f} pts of sack risk")
    print(f"  => EPA via the sack channel {sack_rate_delta * sack_cost:+.4f}")

    net = blitzed["epa"].mean() - not_blitzed["epa"].mean()
    boot_net = cluster_bootstrap(
        df, lambda s: s[s.blitz == 1]["epa"].mean() - s[s.blitz == 0]["epa"].mean())
    lo, hi = np.percentile(boot_net, [2.5, 97.5])
    print(f"\n  NET EPA effect of a blitz   {net:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P(blitz hurts the offense)  {(boot_net < 0).mean():.1%}")
    print("\n  The net effect is SMALLER than the sack channel alone, because a blitz")
    print("  also gives up coverage. Blitzing was close to EPA-neutral in 2021 -- which")
    print("  caps how much a blitz-warning system can be worth, whatever its AUC.")

    summary = {
        "verdict": "cost ratio NOT identified from observational data",
        "why": ("pff_role == 'Pass Block' is assigned post-snap, so '6+ blockers' "
                "marks max-protect play design (61% play-action, 12.9 air yards) "
                "rather than a pre-snap protection call. The false-alarm cost comes "
                "out wrong-signed."),
        "cost_false_negative_epa": float(cost_fn),
        "cost_false_positive_epa": float(cost_fp),
        "cost_fn_ci95": [float(np.percentile(boot_fn, 2.5)), float(np.percentile(boot_fn, 97.5))],
        "cost_fp_ci95": [float(np.percentile(boot_fp, 2.5)), float(np.percentile(boot_fp, 97.5))],
        "epa_cost_of_sack": float(sack_cost),
        "sack_rate_blitz": float(blitzed["sack"].mean()),
        "sack_rate_no_blitz": float(not_blitzed["sack"].mean()),
        "epa_via_sack_channel": float(sack_rate_delta * sack_cost),
        "net_epa_effect_of_blitz": float(net),
        "net_epa_ci95": [float(lo), float(hi)],
        "p_blitz_hurts_offense": float((boot_net < 0).mean()),
        "what_would_identify_it": (
            "Pre-snap protection intent, not post-snap role. The week*.csv tracking "
            "files give every player's position before the snap, so backfield "
            "alignment could stand in for the protection call independently of what "
            "happened after it. That is the experiment this analysis needs."),
        "data_sources": ["NFL Big Data Bowl 2023 (Kaggle)",
                         "nflverse play_by_play_2021 (EPA)"],
    }
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "cost_ratio_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'cost_ratio_analysis.json'}")


if __name__ == "__main__":
    main()
