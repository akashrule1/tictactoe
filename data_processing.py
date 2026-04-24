"""Data engine for cricket T20 prediction system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class DataProcessor:
    """Convert ball-by-ball cricket data into over-level training snapshots."""

    def __init__(
        self,
        processed_dataset_path: str = "data/processed_dataset.csv",
        min_over: float = 1.0,
    ) -> None:
        self.processed_dataset_path = Path(processed_dataset_path)
        self.processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_over = min_over

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    @staticmethod
    def _resolve_column(df: pd.DataFrame, aliases: List[str], required: bool = True) -> str | None:
        for alias in aliases:
            if alias in df.columns:
                return alias
        if required:
            raise ValueError(f"Missing required columns. Expected one of: {aliases}")
        return None

    def load_ball_by_ball_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = self._normalize_columns(df)
        LOGGER.info("Loaded %s rows from %s", len(df), csv_path)
        return df

    def convert_ball_by_ball_to_snapshots(self, ball_df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw ball-level rows to per-over training rows.

        Expected flexible schema aliases:
        - innings: innings, inning
        - over: over, overs
        - ball: ball, ball_in_over
        - total_runs for delivery: runs_off_bat + extras or total_runs
        - wicket indicator: is_wicket or wicket
        """
        df = self._normalize_columns(ball_df)

        innings_col = self._resolve_column(df, ["innings", "inning"])
        over_col = self._resolve_column(df, ["over", "overs"])
        ball_col = self._resolve_column(df, ["ball", "ball_in_over", "ballnumber"], required=False)
        total_runs_col = self._resolve_column(df, ["total_runs", "runs_total", "runs_this_ball"], required=False)
        runs_off_bat_col = self._resolve_column(df, ["runs_off_bat", "batsman_runs"], required=False)
        extras_col = self._resolve_column(df, ["extras", "extra_runs"], required=False)
        wicket_col = self._resolve_column(df, ["is_wicket", "wicket", "player_dismissed"], required=False)
        match_id_col = self._resolve_column(df, ["match_id", "id", "game_id"], required=False)

        if total_runs_col is None:
            if runs_off_bat_col is None:
                raise ValueError("Need either total_runs or runs_off_bat column")
            extras_series = df[extras_col] if extras_col else 0
            df["total_runs"] = df[runs_off_bat_col].fillna(0) + pd.to_numeric(extras_series, errors="coerce").fillna(0)
            total_runs_col = "total_runs"

        if ball_col is None:
            df["ball"] = 1
            ball_col = "ball"

        if wicket_col is None:
            df["is_wicket"] = 0
            wicket_col = "is_wicket"
        else:
            if wicket_col == "player_dismissed":
                df["is_wicket"] = (~df[wicket_col].isna()).astype(int)
                wicket_col = "is_wicket"
            else:
                df[wicket_col] = pd.to_numeric(df[wicket_col], errors="coerce").fillna(0).astype(int)

        if match_id_col is None:
            df["match_id"] = 0
            match_id_col = "match_id"

        df[over_col] = pd.to_numeric(df[over_col], errors="coerce")
        df[ball_col] = pd.to_numeric(df[ball_col], errors="coerce").fillna(1)
        df[total_runs_col] = pd.to_numeric(df[total_runs_col], errors="coerce").fillna(0)

        snapshots: List[Dict] = []

        group_cols = [match_id_col, innings_col]
        for (match_id, innings), g in df.groupby(group_cols, dropna=False):
            g = g.sort_values(by=[over_col, ball_col]).copy()
            g["runs_cum"] = g[total_runs_col].cumsum()
            g["wkts_cum"] = g[wicket_col].cumsum()

            final_score = float(g["runs_cum"].iloc[-1])

            over_agg = (
                g.groupby(over_col)
                .agg(
                    over_runs=(total_runs_col, "sum"),
                    over_wkts=(wicket_col, "sum"),
                    runs_so_far=("runs_cum", "max"),
                    wickets=("wkts_cum", "max"),
                )
                .reset_index()
                .sort_values(over_col)
            )

            over_agg["last_over_runs"] = over_agg["over_runs"].shift(1).fillna(0)
            over_agg["overs"] = over_agg[over_col] + 1.0
            over_agg["run_rate"] = over_agg["runs_so_far"] / over_agg["overs"].replace(0, np.nan)
            over_agg["run_rate"] = over_agg["run_rate"].fillna(0)
            over_agg["phase"] = over_agg["overs"].apply(self._phase_bucket)
            over_agg["momentum"] = self._momentum(over_agg["last_over_runs"], over_agg["run_rate"])
            over_agg["wicket_impact"] = self._wicket_impact(over_agg["wickets"])
            over_agg["phase_factor"] = over_agg["phase"].map({"powerplay": 1.05, "middle": 1.0, "death": 1.12})
            over_agg["final_score"] = final_score
            over_agg["match_id"] = match_id
            over_agg["innings"] = innings

            over_agg = over_agg[over_agg["overs"] >= self.min_over]

            snapshots.extend(over_agg.to_dict(orient="records"))

        snapshot_df = pd.DataFrame(snapshots)
        if snapshot_df.empty:
            LOGGER.warning("No snapshots created. Check source data.")
            return snapshot_df

        columns = [
            "match_id",
            "innings",
            "runs_so_far",
            "overs",
            "wickets",
            "run_rate",
            "last_over_runs",
            "phase",
            "momentum",
            "wicket_impact",
            "phase_factor",
            "final_score",
        ]
        return snapshot_df[columns]

    @staticmethod
    def _phase_bucket(over_number: float) -> str:
        if over_number <= 6:
            return "powerplay"
        if over_number <= 15:
            return "middle"
        return "death"

    @staticmethod
    def _momentum(last_over_runs: pd.Series, run_rate: pd.Series) -> pd.Series:
        baseline = run_rate.fillna(0)
        return (0.7 * last_over_runs.fillna(0)) + (0.3 * baseline)

    @staticmethod
    def _wicket_impact(wickets: pd.Series) -> pd.Series:
        return np.exp(-0.18 * wickets.fillna(0))

    def append_processed_data(self, snapshot_df: pd.DataFrame) -> Path:
        if snapshot_df.empty:
            LOGGER.info("Snapshot data empty, skipping append.")
            return self.processed_dataset_path

        if self.processed_dataset_path.exists():
            existing = pd.read_csv(self.processed_dataset_path)
            combined = pd.concat([existing, snapshot_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["match_id", "innings", "overs", "runs_so_far", "wickets"],
                keep="last",
            )
        else:
            combined = snapshot_df.copy()

        combined.to_csv(self.processed_dataset_path, index=False)
        LOGGER.info("Processed dataset saved to %s with %s rows", self.processed_dataset_path, len(combined))
        return self.processed_dataset_path

    def load_processed_dataset(self) -> pd.DataFrame:
        if not self.processed_dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {self.processed_dataset_path}")
        return pd.read_csv(self.processed_dataset_path)
