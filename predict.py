"""Prediction CLI for live cricket score forecasting."""

from __future__ import annotations

import argparse
import json

from model import CricketPredictionModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict final score for T20 innings")
    parser.add_argument("--runs", type=float, required=True)
    parser.add_argument("--overs", type=float, required=True)
    parser.add_argument("--wickets", type=float, required=True)
    parser.add_argument("--last-over-runs", type=float, required=True)
    parser.add_argument("--target", type=float, default=None)
    args = parser.parse_args()

    model = CricketPredictionModel()
    if not model.load():
        raise FileNotFoundError("No trained model found. Run train.py first.")

    result = model.predict(
        runs_so_far=args.runs,
        overs=args.overs,
        wickets=args.wickets,
        last_over_runs=args.last_over_runs,
        target=args.target,
    )

    print(
        json.dumps(
            {
                "predicted_final_score": result.predicted_final_score,
                "score_range": list(result.score_range),
                "confidence": result.confidence,
                "yes_no": result.yes_no_prediction,
                "yes_probability": result.yes_probability,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
