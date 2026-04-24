"""Training pipeline for cricket T20 prediction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_processing import DataProcessor
from model import CricketPredictionModel

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("train")


def run_training(
    csv_path: str,
    retrain_min_new_rows: int = 40,
    force: bool = False,
) -> None:
    processor = DataProcessor()
    model = CricketPredictionModel()

    ball_df = processor.load_ball_by_ball_csv(csv_path)
    snapshots = processor.convert_ball_by_ball_to_snapshots(ball_df)
    processor.append_processed_data(snapshots)

    dataset = processor.load_processed_dataset()

    metadata = model.metadata()
    previous_rows = metadata.get("metrics", {}).get("rows", 0)
    new_rows = len(dataset) - previous_rows

    if not force and previous_rows > 0 and new_rows < retrain_min_new_rows:
        LOGGER.info(
            "Skipping training: only %s new rows (threshold=%s)",
            new_rows,
            retrain_min_new_rows,
        )
        return

    metrics = model.train(dataset)
    LOGGER.info("Model trained. Metrics: %s", metrics)


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train cricket predictor from ball-by-ball CSV")
    parser.add_argument("--csv", required=True, help="Path to ball-by-ball CSV")
    parser.add_argument("--min-new-rows", type=int, default=40)
    parser.add_argument("--force", action="store_true", help="Force training regardless of new data threshold")
    return parser


if __name__ == "__main__":
    args = _arg_parser().parse_args()
    if not Path(args.csv).exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    run_training(args.csv, retrain_min_new_rows=args.min_new_rows, force=args.force)
