# AI-Powered T20 Cricket Prediction System

Production-ready Flask + ML system for IPL-style T20 innings forecasting with:
- Final score prediction (with min-max range)
- Short-term YES/NO outcome support via target threshold
- Live re-prediction workflow (submit current innings state at any time)
- Continuous data ingestion + controlled auto-retraining
- Model versioning, metrics, logging, and prediction history dashboard

---

## 1) Project Structure

```text
/workspace/tictactoe
├── app.py                         # Flask web app
├── data_processing.py             # Ball-by-ball -> snapshot conversion engine
├── model.py                       # ML models, feature engineering, save/load, inference
├── train.py                       # Training/auto-learning pipeline CLI
├── predict.py                     # Prediction CLI
├── requirements.txt
├── data/
│   ├── processed_dataset.csv      # Auto-created training dataset
│   ├── prediction_history.csv     # Auto-created dashboard history
│   └── uploads/                   # Uploaded CSV files
├── models/
│   ├── latest_model.joblib        # Latest model pointer file
│   ├── model_<timestamp>.joblib   # Versioned model files
│   └── metadata.json              # Training metadata + metrics
├── templates/
│   └── index.html                 # Mobile-friendly UI
├── static/
│   └── styles.css
└── samples/
    └── ball_by_ball_sample.csv    # Sample format
```

---

## 2) Core AI Design

### Regression task
Predict final innings score (`final_score`) from live state:
- `runs_so_far`
- `overs`
- `wickets`
- `run_rate`
- `last_over_runs`
- `momentum`
- `wicket_impact`
- `phase_factor`

Model: `RandomForestRegressor`

### Classification task
Binary support for YES/NO scenarios (e.g., “Will innings reach X?”):
- Train label: `final_score >= threshold` (default threshold = 160 in training)
- Combine classifier probability with score-distance logistic calibration for robust live YES probability.

Model: `RandomForestClassifier`

### Advanced logic included
- **Momentum** = `0.7 * last_over_runs + 0.3 * run_rate`
- **Wicket impact scaling** = `exp(-0.18 * wickets)`
- **Phase-based adjustment** (`phase_factor`):
  - Powerplay (≤6): 1.05
  - Middle (6–15): 1.00
  - Death (>15): 1.12
- **Prediction range** from tree-wise dispersion (approx. 80% interval)

---

## 3) Data Engine

### Accepted input
CSV ball-by-ball IPL-style data. Flexible column aliases are supported:
- innings: `innings` or `inning`
- over: `over` or `overs`
- ball: `ball`, `ball_in_over`, or `ballnumber`
- runs on delivery: `total_runs` OR (`runs_off_bat` + `extras`)
- wicket flag: `is_wicket`, `wicket`, or `player_dismissed`
- match id: `match_id`, `id`, or `game_id`

### Snapshot conversion logic
`data_processing.py` converts ball-level rows to over-level snapshots:
- each over becomes one training row
- outputs:
  - `runs_so_far`
  - `overs`
  - `wickets`
  - `run_rate`
  - `last_over_runs`
  - `phase`
  - `momentum`
  - `wicket_impact`
  - `phase_factor`
  - `final_score`

Processed rows are appended to `data/processed_dataset.csv` with deduplication.

---

## 4) Continuous Learning / Auto-Retraining

Implemented in `train.py`:
1. Load new match CSV
2. Convert + append snapshots
3. Check new row growth vs previous training metadata
4. Retrain **only** if enough new rows are present (default 40)
5. Save versioned model + update `latest_model.joblib`

Use `--force` to bypass threshold.

---

## 5) Run Locally

## Prerequisites
- Python 3.10+

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model from CSV
```bash
python train.py --csv samples/ball_by_ball_sample.csv --force
```

## Predict from CLI
```bash
python predict.py --runs 82 --overs 10.2 --wickets 3 --last-over-runs 11 --target 170
```

## Run Flask app
```bash
python app.py
```
Open: `http://127.0.0.1:5000`

---

## 6) UI Features

Web app (`app.py`) provides:
- Inputs: `runs`, `overs`, `wickets`, `last_over_runs`, optional `target`
- Buttons:
  - Predict
  - Upload CSV + Convert Data
  - Train Model
- Outputs:
  - Predicted score
  - Min-max range
  - Confidence %
  - YES/NO + YES probability (if target given)
- Dashboard:
  - Latest 25 predictions stored in `data/prediction_history.csv`

---

## 7) Sample Dataset Format

Example (`samples/ball_by_ball_sample.csv`):
```csv
match_id,innings,over,ball,total_runs,is_wicket
1001,1,0,1,1,0
1001,1,0,2,0,0
1001,1,0,3,4,0
...
```

---

## 8) APK / Mobile Deployment

### Preferred method (WebView wrapper for Flask-hosted UI)

1. Deploy Flask app to a reachable endpoint (Render/Railway/EC2/etc.) or LAN host.
2. Create Android WebView shell (Android Studio):
   - New Empty Activity
   - Add INTERNET permission in `AndroidManifest.xml`
   - Load your app URL in WebView
3. Build signed APK from Android Studio (`Build > Generate Signed Bundle / APK`).

### Alternative: Convert to Kivy + Buildozer (native APK)

1. Port UI to Kivy screens (reuse prediction/training APIs from `model.py`/`data_processing.py`).
2. On Linux:
   ```bash
   pip install buildozer cython
   buildozer init
   ```
3. Edit `buildozer.spec`:
   - set app title/package
   - add requirements: `python3,flask,pandas,numpy,scikit-learn,joblib`
4. Build:
   ```bash
   buildozer -v android debug
   ```
5. APK generated under `bin/`.

> For production mobile, WebView + hosted backend is usually simpler and lighter than embedding model training on-device.

---

## 9) Extensibility Ideas

- Swap RandomForest for XGBoost/LightGBM for improved structured-data performance
- Add batter/bowler/team embeddings
- Add win-probability model for chase context
- Add websocket feed for real-time live ball updates
- Add drift detection before retraining

