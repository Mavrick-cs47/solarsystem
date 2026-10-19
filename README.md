# SolarSight AI — solar forecasting and panel-condition inspection

Modular add-on for a GitHub repository. It contains:

| Part | What it does |
|---|---|
| `backend/app/data`, `forecast` | Data cleaning, deterministic synthetic data generator, baseline vs Random Forest vs Gradient Boosting, chronological split, MAE/RMSE/R², saved model loaded once at startup |
| `backend/app/vision` | OpenCV preprocessing, upload validation, MobileNetV2 transfer-learning training script (Keras), inference with class probabilities |
| `backend/app/assistant` | Gemini explanation (backend only) + deterministic fallback + local guidance retrieval with source references |
| `backend/app/main.py` | FastAPI routes under `/api` |
| `frontend/` | React + Vite dashboard (forecast chart, inputs, image upload, assistant, CSV / printable report) |
| `.github/workflows/ci.yml`, `docker-compose.yml`, `*/Dockerfile` | CI and containers |

## Adding it to your existing repository

1. Create a branch: `git checkout -b add-solarsight`.
2. Copy the folders from this ZIP into the repo root. Nothing in it needs to sit at a special path except what the files reference:
   `backend/`, `frontend/`, `data/`, `.github/workflows/ci.yml`, `docker-compose.yml`, `.env.example`, `.gitignore`.
3. **Merge, don't overwrite**, if your repo already has the same file names. The likely clashes are `.gitignore`,
   `.env.example`, `.github/workflows/ci.yml`, `docker-compose.yml`, and anything under `backend/app/` or `frontend/src/`.
   Compare them first (`git diff`) and keep your own settings.
4. If your repo already has a backend or frontend you want to keep, you can use just the pieces you need. The modules are
   independent: `app/forecast/*` and `app/data/*` need only pandas/scikit-learn; `app/vision/preprocess.py` and `dataset.py`
   need only OpenCV/Pillow/scikit-learn; TensorFlow is imported lazily and only by `vision/train.py` and `vision/inference.py`.
5. Never commit `backend/.env`. `.gitignore` already excludes it, trained models and dataset images.

## Quick start (Windows PowerShell)

Requires Python 3.11 or newer (developed on 3.13; CI is set to 3.12) and Node 20+.

```powershell
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy ..\.env.example .env        # optional: add GEMINI_API_KEY here
uvicorn app.main:app --reload --port 8000
```

If PowerShell blocks the activation script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

On first start the backend trains the forecast model once on synthetic data (about 20 seconds) and saves it to
`backend/models/`. Later starts load the saved file.

```powershell
# Frontend (second terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to port 8000.

### Docker

```powershell
copy .env.example .env     # optional
docker compose up --build
```

Open http://localhost:8080. Set `INSTALL_VISION: "true"` in `docker-compose.yml` to include TensorFlow.

## Training

### Forecast model on your own data

```powershell
cd backend
python -m app.forecast.training --csv path\to\history.csv
```

Required columns: `timestamp, temperature_c, cloud_cover_pct, humidity_pct, wind_speed_ms, irradiance_wm2, power_kw`.
Optional: `capacity_kw` (default 100). Invalid values (for example cloud cover above 100 or negative power) are blanked and
counted, duplicate timestamps are dropped, gaps of up to 3 hours in weather columns are interpolated, and rows missing the
target are dropped. The cleaning report is stored in the model metadata. Without `--csv`, deterministic synthetic data is used.

### Image classifier (needs a dataset and TensorFlow)

```powershell
pip install -r requirements-vision.txt
cd ..                                  # repo root
# put images in data\panel_images\<class_name>\*.jpg  (see data\panel_images\README.md)
cd backend
python -m app.vision.train --dataset ..\data\panel_images --epochs 10
```

Images are split 70/15/15 (stratified) into train/validation/test. The script prints accuracy, per-class precision, recall,
F1 and the confusion matrix on the **test** split, then writes `backend/models/panel_classifier.keras` and
`panel_classifier_labels.json`. `--no-pretrained` trains from scratch without downloading ImageNet weights; use it only for
offline smoke tests, because results on small datasets will be poor.

## Environment variables

See `.env.example`. The important ones: `GEMINI_API_KEY` (backend only; without it the assistant uses the deterministic
fallback), `MAX_UPLOAD_MB`, `CORS_ORIGINS`, `AUTO_TRAIN_FORECAST`, `MODEL_DIR`. Variables starting with `VITE_` are sent to
the browser, so never put secrets in them.

## API

Interactive docs at http://localhost:8000/docs. Errors share one shape: `{"error": {"code": "...", "message": "..."}}`
(validation errors use FastAPI's standard 422 `detail` list).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, whether a forecast model file exists, vision model availability, whether Gemini is configured |
| POST | `/api/forecast` | Hourly forecast for one day. Body: `date, capacity_kw, temperature_c, cloud_cover_pct, humidity_pct, wind_speed_ms`, optional `hourly_weather` (exactly 24 entries) |
| GET | `/api/model/metrics` | Validation/test metrics for all candidate models, baseline comparison, split sizes, cleaning report |
| GET | `/api/inspect/status` | Whether the image classifier is available and its test metrics |
| POST | `/api/inspect` | Multipart field `file` (JPG/PNG/WEBP, up to `MAX_UPLOAD_MB`). Returns class, confidence, probabilities. `503 vision_model_missing` if no model is trained |
| POST | `/api/assistant/explain` | Body: `forecast` and/or `inspection` (the objects returned by the endpoints above) plus optional `question`. Returns `summary, observed_facts, model_predictions, uncertainty, suggested_next_steps, retrieved_sources, requires_human_review, generated_by, fallback_reason` |

Example:

```powershell
curl.exe -X POST http://localhost:8000/api/forecast -H "Content-Type: application/json" `
  -d '{\"date\":\"2025-06-15\",\"capacity_kw\":100,\"temperature_c\":30,\"cloud_cover_pct\":20,\"humidity_pct\":45,\"wind_speed_ms\":3}'
```

## Design decisions worth knowing (for interviews)

- **Leakage control:** the split is chronological (train → validation → test). The model is chosen on validation only,
  refit on train+validation, and scored once on the untouched test block. Features use only inputs available at forecast time
  (solar geometry and weather); no lagged or future target values.
- **Honest metrics:** power errors are in % of installed capacity. Night hours have zero output and inflate R², so
  daytime-only metrics are reported too, next to the linear baseline.
- **Uncertainty:** the shaded band is the empirical 5th–95th percentile of test residuals. It describes past error, not a guarantee.
- **Assistant safety:** Gemini receives only the supplied results and retrieved notes, must return JSON matching a schema,
  and any failure (no key, timeout, rate limit, malformed output) falls back to a template built from the same evidence.
  `requires_human_review` is always true. Retrieved guidance is shown separately from generated text.
- **Upload safety:** extension, size, file signature (magic bytes), pixel count and decodability are all checked.

## Limitations and what works when

| Works immediately | Needs something from you |
|---|---|
| Forecast API and dashboard (trained on synthetic data on first start) | Real forecast accuracy: needs your real historical CSV (`--csv`) |
| Metrics, report, CSV export, print view | Panel image classification: needs a labelled dataset you are licensed to use, TensorFlow, and a training run. **No trained weights are included** |
| Assistant fallback (no key) | Gemini wording: needs `GEMINI_API_KEY` |
| Tests and CI | Docker images: not built in the environment this was written in (see below) |

- The synthetic data is generated from simple physical formulas. Metrics on it (R² around 0.99) show the pipeline works,
  **not** that the model would be this accurate on a real plant. Do not quote them as real accuracy.
- Inputs are a single-day scenario; the project does not fetch live weather forecasts.
- Solar position uses an approximate formula and treats timestamps as local solar time.
- The guidance notes in `backend/app/assistant/knowledge.py` are short project-written summaries, not official documents.

## Test report (what was actually run)

Run on Linux with Python 3.13 and Node 22 while building this:

- Backend: `python -m pytest` → **37 passed**, using the pinned `requirements-dev.txt` in a clean virtualenv
  (pandas 2.3.3, scikit-learn 1.9.1, FastAPI 0.146). Also passed on pandas 3.0.5.
- Frontend: `npm test` → 4 passed; `npm run build` → succeeded (bundle ≈ 566 kB, Vite warns about chunk size).
- Live check: backend served real `/api/health` and `/api/forecast` responses; the dashboard was loaded in headless Chromium
  with no console errors, and the assistant tab produced a fallback answer.
- Vision pipeline: trained for 3 epochs with `--no-pretrained` on 120 *generated* coloured images (TensorFlow 2.21) to confirm
  that data splitting, training, evaluation, saving, loading and `/api/inspect` work. Accuracy on that toy data was 33%, which
  is meaningless; this only proves the code path. The ImageNet-weights path could not be tested because the sandbox blocks the download.
- Not run: Docker image builds (no Docker daemon available; `docker compose config` validated the compose file's syntax only),
  the GitHub Actions workflow itself, a real Gemini call (the tests mock the HTTP layer), and anything on Windows.

## Troubleshooting

- **`503 vision_model_missing`** — expected until you train a classifier (see Training).
- **`503 tensorflow_not_installed`** — `pip install -r requirements-vision.txt`.
- **Frontend says it cannot reach the API** — the backend must be running on port 8000; in production set `VITE_API_BASE` or use the nginx proxy from `docker compose`.
- **First backend start is slow** — it is training the demo model once; set `AUTO_TRAIN_FORECAST=false` and run the training command yourself if you prefer.
- **`Could not download the ImageNet weights`** — you need internet access for the pretrained path, or use `--no-pretrained`.
- **Loaded model complains about scikit-learn versions** — delete `backend/models/` and retrain with the same environment.
