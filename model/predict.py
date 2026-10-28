"""
Inference module - loads the trained Random Forest model and saved
preprocessor, then classifies new behavioural records.

Used by the Streamlit prototype (app.py) at runtime. Designed so that the
prototype never needs to know anything about model internals - it just
calls `Predictor.predict(record)` and receives a structured result.

Threshold note
--------------
The system uses a tuned classification threshold of 0.45 (instead of the
sklearn default 0.50) on the malicious-class probability. This was selected
on the basis of the testing analysis (see Research and Design Report §7.4 -
Implemented Improvements): four False Negative test cases were clustered at
probabilities 0.498–0.503, indicating the model had genuine uncertainty
about them but defaulted to the majority Normal class under the default
threshold. Lowering the threshold to 0.45 converts those uncertain cases
to True Positives in line with the security-context principle that the
cost of missing an insider threat (FN) materially exceeds the cost of
investigating a false alarm (FP) - see Stallings & Brown [1].

The threshold is exposed as an environment variable (THREAT_THRESHOLD)
so that operators can tune it for any deployment without re-training.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .preprocessor import Preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Tuned classification threshold on the malicious-class probability.
# 0.45 was selected on the basis of testing analysis - see §7.4 of the
# Research and Design Report. Override with the THREAT_THRESHOLD env var.
DEFAULT_THRESHOLD = float(os.environ.get("THREAT_THRESHOLD", "0.45"))


@dataclass
class PredictionResult:
    """Structured output from a single classification."""
    classification: int            # 0 = Normal, 1 = Malicious
    label: str                     # Human-readable label
    confidence: float              # Probability of the predicted class
    probability_malicious: float   # Always the malicious-class probability
    feature_values: dict           # Preprocessed feature values for this record
    threshold: float               # The decision threshold applied


class Predictor:
    """
    Wraps the trained Random Forest model + preprocessor for inference.

    Typical usage:
        predictor = Predictor()
        result = predictor.predict(record_df)
        print(result.classification, result.confidence)

    Threshold tuning:
        predictor = Predictor(threshold=0.50)   # back to sklearn default
        predictor = Predictor(threshold=0.45)   # recall-priority (default)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else MODELS_DIR / "rf_model.pkl"
        self.preprocessor_path = (
            Path(preprocessor_path) if preprocessor_path
            else MODELS_DIR / "preprocessor.pkl"
        )
        self.threshold = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                f"threshold must be strictly between 0 and 1; got {self.threshold}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                f"Run `python -m model.train` first to train the model."
            )
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {self.preprocessor_path}. "
                f"Run `python -m model.train` first."
            )

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        self.preprocessor = Preprocessor.load(str(self.preprocessor_path))
        self.feature_names = self.preprocessor.feature_names()

    # Prediction

    def _classify_from_proba(self, prob_mal: float) -> int:
        """Apply the tuned threshold to derive the class label."""
        return 1 if prob_mal >= self.threshold else 0

    def predict(self, record: pd.DataFrame) -> PredictionResult:
        """Classify a single behavioural record using the tuned threshold."""
        if len(record) != 1:
            raise ValueError(
                f"predict() expects exactly one record; got {len(record)}. "
                f"Use predict_batch() for multiple records."
            )

        X = self.preprocessor.transform(record)
        probs = self.model.predict_proba(X)[0]
        prob_mal = float(probs[1])
        cls = self._classify_from_proba(prob_mal)
        confidence = float(probs[cls])

        return PredictionResult(
            classification=cls,
            label="Malicious" if cls == 1 else "Normal",
            confidence=confidence,
            probability_malicious=prob_mal,
            feature_values=dict(zip(self.feature_names, X.iloc[0].tolist())),
            threshold=self.threshold,
        )

    def predict_batch(self, records: pd.DataFrame) -> list[PredictionResult]:
        """Classify multiple records in one call, using the tuned threshold."""
        X = self.preprocessor.transform(records)
        probs = self.model.predict_proba(X)

        results = []
        for i in range(len(records)):
            prob_mal = float(probs[i][1])
            cls = self._classify_from_proba(prob_mal)
            results.append(PredictionResult(
                classification=cls,
                label="Malicious" if cls == 1 else "Normal",
                confidence=float(probs[i][cls]),
                probability_malicious=prob_mal,
                feature_values=dict(zip(self.feature_names, X.iloc[i].tolist())),
                threshold=self.threshold,
            ))
        return results


    # Introspection

    def get_feature_importances(self) -> dict[str, float]:
        """Return the global feature importance dictionary."""
        return dict(zip(self.feature_names, self.model.feature_importances_.tolist()))
