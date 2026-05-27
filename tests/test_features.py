"""Unit tests for feature engineering and label construction in train_model.py.

These tests use small in-memory DataFrames so they don't depend on the
actual BDB data being present.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make training/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from train_model import (  # noqa: E402
    BLITZ_THRESHOLD,
    build_blitz_labels,
    engineer_features,
    gameclock_to_seconds,
)


class TestGameClockToSeconds:
    def test_normal_clock(self):
        assert gameclock_to_seconds("14:32") == 14 * 60 + 32

    def test_zero_clock(self):
        assert gameclock_to_seconds("0:00") == 0

    def test_end_of_quarter(self):
        assert gameclock_to_seconds("15:00") == 900

    def test_missing_clock(self):
        assert gameclock_to_seconds(None) == 0

    def test_malformed_clock_returns_zero(self):
        # The function is defensive — bad input shouldn't crash training
        assert gameclock_to_seconds("garbage") == 0


class TestBuildBlitzLabels:
    def _make_pff(self, plays_with_rusher_counts):
        """Helper: build a fake pffScoutingData frame.

        plays_with_rusher_counts: list of (gameId, playId, num_rushers) tuples.
        Each entry produces num_rushers rows with pff_role='Pass Rush' plus
        a couple of coverage rows so the filter actually does something.
        """
        rows = []
        for game_id, play_id, n_rushers in plays_with_rusher_counts:
            for _ in range(n_rushers):
                rows.append({"gameId": game_id, "playId": play_id, "pff_role": "Pass Rush"})
            # Always a couple of coverage defenders to make sure we're filtering
            rows.append({"gameId": game_id, "playId": play_id, "pff_role": "Pass Coverage"})
            rows.append({"gameId": game_id, "playId": play_id, "pff_role": "Pass Coverage"})
        return pd.DataFrame(rows)

    def test_threshold_is_five(self):
        """Sanity check: standard NFL blitz definition is 5+ pass rushers."""
        assert BLITZ_THRESHOLD == 5

    def test_blitz_when_five_rushers(self):
        pff = self._make_pff([(1, 100, 5)])
        result = build_blitz_labels(pff)
        assert len(result) == 1
        assert result.iloc[0]["num_pass_rushers"] == 5
        assert result.iloc[0]["was_blitz"] == 1

    def test_no_blitz_when_four_rushers(self):
        pff = self._make_pff([(1, 100, 4)])
        result = build_blitz_labels(pff)
        assert result.iloc[0]["was_blitz"] == 0

    def test_heavy_blitz_six_rushers(self):
        pff = self._make_pff([(1, 100, 6)])
        assert result_first_value(build_blitz_labels(pff), "was_blitz") == 1

    def test_mixed_plays(self):
        pff = self._make_pff([
            (1, 100, 4),  # no blitz
            (1, 101, 5),  # blitz
            (1, 102, 7),  # heavy blitz
            (2, 200, 3),  # 3-man rush, no blitz
        ])
        result = build_blitz_labels(pff).sort_values(["gameId", "playId"]).reset_index(drop=True)
        assert list(result["num_pass_rushers"]) == [4, 5, 7, 3]
        assert list(result["was_blitz"]) == [0, 1, 1, 0]

    def test_coverage_only_play_dropped(self):
        """A play with no Pass Rush rows shouldn't appear in output at all."""
        pff = pd.DataFrame([
            {"gameId": 1, "playId": 100, "pff_role": "Pass Coverage"},
            {"gameId": 1, "playId": 100, "pff_role": "Pass Coverage"},
        ])
        result = build_blitz_labels(pff)
        assert len(result) == 0


class TestEngineerFeatures:
    def _make_plays(self, **overrides):
        defaults = {
            "gameId": 1,
            "playId": 100,
            "possessionTeam": "KC",
            "defensiveTeam": "BUF",
            "quarter": 1,
            "down": 1,
            "yardsToGo": 10,
            "absoluteYardlineNumber": 75,
            "preSnapHomeScore": 0,
            "preSnapVisitorScore": 0,
            "gameClock": "15:00",
        }
        defaults.update(overrides)
        return pd.DataFrame([defaults])

    def _games(self):
        return pd.DataFrame([
            {"gameId": 1, "homeTeamAbbr": "KC", "visitorTeamAbbr": "BUF"},
        ])

    def test_is_home_offense_true_when_possession_is_home(self):
        plays = self._make_plays(possessionTeam="KC")
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["is_home_offense"] == 1

    def test_is_home_offense_false_when_visiting(self):
        plays = self._make_plays(possessionTeam="BUF")
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["is_home_offense"] == 0

    def test_score_differential_from_offense_perspective(self):
        # Home team possesses, home 21, visitor 14 -> +7 for offense
        plays = self._make_plays(possessionTeam="KC", preSnapHomeScore=21, preSnapVisitorScore=14)
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["score_differential"] == 7

        # Visitor possesses, same scoreboard -> -7 for offense
        plays = self._make_plays(possessionTeam="BUF", preSnapHomeScore=21, preSnapVisitorScore=14)
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["score_differential"] == -7

    def test_game_seconds_remaining_q1_full_clock(self):
        plays = self._make_plays(quarter=1, gameClock="15:00")
        result = engineer_features(plays, self._games())
        # Q1 15:00 = 15min Q1 + 15min Q2 + 30min H2 = 60min = 3600s
        assert result.iloc[0]["game_seconds_remaining"] == 3600

    def test_game_seconds_remaining_q4_end(self):
        plays = self._make_plays(quarter=4, gameClock="0:00")
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["game_seconds_remaining"] == 0

    def test_is_two_minute_q2(self):
        plays = self._make_plays(quarter=2, gameClock="1:45")
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["is_two_minute"] == 1

    def test_is_two_minute_q1_doesnt_count(self):
        # 2-minute warning only matters in Q2 and Q4
        plays = self._make_plays(quarter=1, gameClock="1:45")
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["is_two_minute"] == 0

    def test_is_red_zone(self):
        plays = self._make_plays(absoluteYardlineNumber=15)
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["is_red_zone"] == 1

    def test_not_red_zone_at_midfield(self):
        plays = self._make_plays(absoluteYardlineNumber=50)
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["is_red_zone"] == 0

    def test_is_goal_to_go(self):
        # Ball at the 3, 4 yards to go -> goal-to-go (4 >= 3)
        plays = self._make_plays(absoluteYardlineNumber=3, yardsToGo=4)
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["is_goal_to_go"] == 1

    def test_not_goal_to_go_midfield_first_and_ten(self):
        plays = self._make_plays(absoluteYardlineNumber=50, yardsToGo=10)
        result = engineer_features(plays, self._games())
        assert result.iloc[0]["is_goal_to_go"] == 0


def result_first_value(df: pd.DataFrame, col: str):
    return df.iloc[0][col]
