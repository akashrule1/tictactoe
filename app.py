"""Flask app for AI-powered T20 score prediction."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from data_processing import DataProcessor
from model import CricketPredictionModel

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("app")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = Path("data/prediction_history.csv")

app = Flask(__name__)
app.secret_key = "replace-with-env-secret-for-production"
processor = DataProcessor()
model = CricketPredictionModel()


def _append_history(row: dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    exists = HISTORY_FILE.exists()
    with HISTORY_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "runs",
                "overs",
                "wickets",
                "last_over_runs",
                "target",
                "predicted_final",
                "range_min",
                "range_max",
                "confidence",
                "yes_no",
                "yes_probability",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_history(limit: int = 25):
    if not HISTORY_FILE.exists():
        return []
    rows = []
    with HISTORY_FILE.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return list(reversed(rows[-limit:]))


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "predict":
            if not model.load():
                flash("Model not found. Train a model first.", "warning")
                return redirect(url_for("index"))

            runs = float(request.form["runs"])
            overs = float(request.form["overs"])
            wickets = float(request.form["wickets"])
            last_over_runs = float(request.form["last_over_runs"])
            target_input = request.form.get("target", "").strip()
            target = float(target_input) if target_input else None

            result = model.predict(runs, overs, wickets, last_over_runs, target)
            _append_history(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "runs": runs,
                    "overs": overs,
                    "wickets": wickets,
                    "last_over_runs": last_over_runs,
                    "target": target if target is not None else "",
                    "predicted_final": result.predicted_final_score,
                    "range_min": result.score_range[0],
                    "range_max": result.score_range[1],
                    "confidence": result.confidence,
                    "yes_no": result.yes_no_prediction or "",
                    "yes_probability": result.yes_probability if result.yes_probability is not None else "",
                }
            )

        elif action == "convert":
            upload = request.files.get("csv_file")
            if not upload or not upload.filename:
                flash("Please upload a CSV file first.", "warning")
                return redirect(url_for("index"))

            dst = UPLOAD_DIR / upload.filename
            upload.save(dst)
            ball_df = processor.load_ball_by_ball_csv(str(dst))
            snapshots = processor.convert_ball_by_ball_to_snapshots(ball_df)
            processor.append_processed_data(snapshots)
            flash(f"Converted and appended {len(snapshots)} snapshot rows.", "success")
            return redirect(url_for("index"))

        elif action == "train":
            try:
                dataset = processor.load_processed_dataset()
            except FileNotFoundError:
                flash("No processed dataset found. Upload and convert CSV first.", "warning")
                return redirect(url_for("index"))

            metrics = model.train(dataset)
            flash(f"Model trained: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}", "success")
            return redirect(url_for("index"))

    metadata = model.metadata()
    history = _load_history()
    return render_template("index.html", result=result, metadata=metadata, history=history)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
