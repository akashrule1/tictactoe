"""Model layer for cricket prediction system."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "runs_so_far",
    "overs",
    "wickets",
    "run_rate",
    "last_over_runs",
    "momentum",
    "wicket_impact",
    "phase_factor",
]


@dataclass
class PredictionResult:
    predicted_final_score: int
    score_range: Tuple[int, int]
    confidence: float
    yes_no_prediction: str | None
    yes_probability: float | None


class CricketPredictionModel:
    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.regressor = RandomForestRegressor(
            n_estimators=350,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

        self.model_state_path = self.model_dir / "latest_model.joblib"
        self.meta_path = self.model_dir / "metadata.json"

    @staticmethod
    def _phase_to_factor(phase: str) -> float:
        mapping = {"powerplay": 1.05, "middle": 1.0, "death": 1.12}
        return mapping.get((phase or "middle").lower(), 1.0)

    @staticmethod
    def _phase(overs: float) -> str:
        if overs <= 6:
            return "powerplay"
        if overs <= 15:
            return "middle"
        return "death"

    @staticmethod
    def _wicket_impact(wickets):
        return np.exp(-0.18 * wickets)

    @staticmethod
    def _momentum(last_over_runs: float, run_rate: float) -> float:
        return (0.7 * last_over_runs) + (0.3 * run_rate)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        if "run_rate" not in data.columns:
            data["run_rate"] = data["runs_so_far"] / data["overs"].replace(0, np.nan)
            data["run_rate"] = data["run_rate"].fillna(0)

        if "phase_factor" not in data.columns:
            phase = data["overs"].apply(self._phase)
            data["phase_factor"] = phase.map({"powerplay": 1.05, "middle": 1.0, "death": 1.12})

        if "momentum" not in data.columns:
            data["momentum"] = self._momentum(data["last_over_runs"], data["run_rate"])

        if "wicket_impact" not in data.columns:
            data["wicket_impact"] = self._wicket_impact(data["wickets"])

        return data[FEATURE_COLUMNS]

    def train(self, dataset: pd.DataFrame, retrain_threshold_target: int = 160) -> Dict:
        if len(dataset) < 20:
            raise ValueError("Need at least 20 training rows for stable model performance")

        X = self.prepare_features(dataset)
        y_reg = dataset["final_score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        self.regressor.fit(X_train, y_train)
        reg_preds = self.regressor.predict(X_test)

        y_cls = (dataset["final_score"] >= retrain_threshold_target).astype(int)
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)
        self.classifier.fit(Xc_train, yc_train)
        cls_preds = self.classifier.predict(Xc_test)

        metrics = {
            "rows": int(len(dataset)),
            "mae": float(mean_absolute_error(y_test, reg_preds)),
            "rmse": float(root_mean_squared_error(y_test, reg_preds)),
            "accuracy": float(accuracy_score(yc_test, cls_preds)),
            "f1": float(f1_score(yc_test, cls_preds)),
        }

        self.save(metrics)
        LOGGER.info("Training complete: %s", metrics)
        return metrics

    def predict(
        self,
        runs_so_far: float,
        overs: float,
        wickets: float,
        last_over_runs: float,
        target: float | None = None,
    ) -> PredictionResult:
        run_rate = runs_so_far / overs if overs > 0 else 0
        phase = self._phase(overs)
        row = pd.DataFrame(
            [
                {
                    "runs_so_far": runs_so_far,
                    "overs": overs,
                    "wickets": wickets,
                    "run_rate": run_rate,
                    "last_over_runs": last_over_runs,
                    "momentum": self._momentum(last_over_runs, run_rate),
                    "wicket_impact": self._wicket_impact(wickets),
                    "phase_factor": self._phase_to_factor(phase),
                }
            ]
        )

        y_pred = float(self.regressor.predict(row[FEATURE_COLUMNS])[0])
        tree_preds = np.array([tree.predict(row[FEATURE_COLUMNS])[0] for tree in self.regressor.estimators_])
        std_dev = float(np.std(tree_preds))
        interval_min = max(int(round(y_pred - 1.28 * std_dev)), int(runs_so_far))
        interval_max = int(round(y_pred + 1.28 * std_dev))

        confidence = max(30.0, min(99.0, 100.0 - (std_dev * 1.8)))

        yes_no_prediction = None
        yes_probability = None
        if target is not None:
            target_prob = float(self.classifier.predict_proba(row[FEATURE_COLUMNS])[0][1])
            projected_prob = max(0.0, min(1.0, 0.55 * target_prob + 0.45 * (1 / (1 + np.exp(-(y_pred - target) / 8)))))
            yes_probability = projected_prob
            yes_no_prediction = "YES" if y_pred >= target else "NO"

        return PredictionResult(
            predicted_final_score=int(round(y_pred)),
            score_range=(interval_min, interval_max),
            confidence=round(confidence, 2),
            yes_no_prediction=yes_no_prediction,
            yes_probability=round(100 * yes_probability, 2) if yes_probability is not None else None,
        )

    def save(self, metrics: Dict) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_path = self.model_dir / f"model_{timestamp}.joblib"
        payload = {
            "regressor": self.regressor,
            "classifier": self.classifier,
            "features": FEATURE_COLUMNS,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(payload, version_path)
        joblib.dump(payload, self.model_state_path)

        meta = {
            "latest_version": version_path.name,
            "last_trained": payload["saved_at_utc"],
            "metrics": metrics,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def load(self) -> bool:
        if not self.model_state_path.exists():
            return False
        payload = joblib.load(self.model_state_path)
        self.regressor = payload["regressor"]
        self.classifier = payload["classifier"]
        return True

    def metadata(self) -> Dict:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text())
