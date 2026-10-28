# Insider Threat Detection with Explainable ML

A prototype that flags employees whose activity looks like an insider threat, and explains every
decision in plain English so a manager can see why a record was flagged.

Computer and Information Security project (COS720), University of Pretoria, 2026.

## Results

Trained on 118,614 employee-activity records, of which about 5% are malicious, with 20% held out for
testing (23,723 records, 1,277 malicious). Test-set results for the malicious class:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| **Random Forest** | **0.68** | **0.82** | **0.74** |
| Decision tree | 0.47 | 0.80 | 0.59 |
| Linear SVM | 0.29 | 0.73 | 0.42 |
| Logistic regression | 0.27 | 0.75 | 0.40 |

These figures use the default 0.5 decision threshold. The app applies a tuned threshold of 0.45, which
favours catching threats over avoiding false alarms; set the `THREAT_THRESHOLD` environment variable to
change it. Confusion matrix, feature importance and model comparison charts are in [`reports/`](reports/).

## The app

A Streamlit interface with two tabs:

- **Analysis:** upload a CSV. One row gives a per-record explanation; many rows give a batch dashboard
  you can drill into.
- **Model & About:** how the Random Forest behaves overall, plus the framing, ethics and limitations.

`data/sample_profiles.csv` has five demo profiles to try.

## Data

The training dataset was supplied for the module and isn't included here. To retrain, place it at
`data/insider_threat_clean_dataset.csv`. It has the same columns as `data/sample_profiles.csv`
(without `profile_name`), plus the label column `is_malicious`.

## Run locally

Needs Python 3.10 or newer.

```bash
pip install -r requirements.txt
python -m model.train
streamlit run app.py
```

Then open http://localhost:8501.

## Run with Docker

```bash
docker build -t insider-threat-detection .
docker run --rm -p 8501:8501 -v "$(pwd)/models:/app/models" -v "$(pwd)/reports:/app/reports" insider-threat-detection serve
```

`serve` trains the models first if none exist, so put the dataset in `data/` before building. The
container also accepts `train`, `predict` and `shell`.

## Project structure

```
model/
  preprocessor.py   encoding and feature engineering, shared by training and inference
  train.py          trains all four models and writes metrics and charts
  predict.py        inference wrapper that applies the decision threshold
  explainer.py      feature attribution and plain-English explanations
app.py              Streamlit app
data/               demo profiles (the training data goes here too)
models/             trained models, created by train.py
reports/            metrics.json and charts
```

## Built with

Python, scikit-learn, pandas, Streamlit, Plotly, matplotlib, Docker
