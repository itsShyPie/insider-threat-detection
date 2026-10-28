"""
Shared preprocessing logic for the AI-Powered Insider Threat Detection System.

Used by both `train.py` (during model training) and `predict.py` (during
inference in the Streamlit prototype). Using the same preprocessor in both
contexts guarantees that the transformations applied at inference time exactly
match those applied during training - a key requirement for production ML
systems (NFR-6 in the Requirements Specification).

This module:
  - Encodes the four categorical features with sklearn LabelEncoder objects
    that are persisted to disk.
  - Engineers two derived features that surface behavioural patterns directly
    relevant to insider threat detection.
  - Provides save/load methods for round-tripping the fitted encoders.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Columns used by the model. Order matters - predict.py relies on this.
CATEGORICAL_COLS: list[str] = [
    "employee_department",
    "employee_campus",
    "employee_position",
    "employee_origin_country",
]

NUMERIC_COLS: list[str] = [
    "employee_seniority_years",
    "is_contractor",
    "employee_classification",
    "has_foreign_citizenship",
    "has_criminal_record",
    "has_medical_history",
    "total_printed_pages",
    "num_printed_pages_off_hours",
    "total_files_burned",
    "burned_from_other",
    "is_abroad",
    "trip_day_number",
    "hostility_country_level",
    "num_entries",
    "num_unique_campus",
    "late_exit_flag",
    "entry_during_weekend",
]

TARGET_COL: str = "is_malicious"

# Derived (engineered) feature names. Computed inside engineer_features.
DERIVED_COLS: list[str] = [
    "off_hours_print_ratio",
    "files_burned_per_entry",
]


@dataclass
class Preprocessor:
    """
    Stateful preprocessor for the insider threat dataset.

    Holds fitted LabelEncoder objects (one per categorical column) and exposes
    `fit_transform` for training and `transform` for inference. The instance is
    designed to be pickled via `save()` after training and reloaded via
    `load()` at inference time.
    """

    encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    fitted: bool = False

    # Feature engineering

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features from the raw record(s).

        Two features are engineered:
          - off_hours_print_ratio: the proportion of printing activity that
            occurred outside business hours. A high ratio is a strong
            behavioural indicator of malicious printing patterns.
          - files_burned_per_entry: removable-media write activity normalised
            by physical presence. Captures users who burn disproportionately
            many files relative to how often they are on site.

        Adding 1 to denominators prevents division-by-zero for users with no
        prints or no entries on record.
        """
        # Defensive copy so we never mutate the input DataFrame.
        df = df.copy()

        # Engineered feature 1: proportion of employee print activity
        # that happened outside business hours. A small value means most
        # printing was during the day; a value close to 1 means almost all
        # printing was off-hours - a behavioural red flag.
        # The "+ 1" in the denominator avoids division-by-zero for
        # employees who didn't print at all.
        df["off_hours_print_ratio"] = (
            df["num_printed_pages_off_hours"] / (df["total_printed_pages"] + 1)
        )

        # Engineered feature 2: removable-media writes per facility entry.
        # Normalises file-burn volume by how often the employee was on site.
        # An employee who is rarely on site but burns many files looks
        # different from one whose burns are spread across many visits.
        df["files_burned_per_entry"] = (
            df["total_files_burned"] / (df["num_entries"] + 1)
        )

        return df

    # Categorical encoding

    def fit_encoders(self, df: pd.DataFrame) -> None:
        """Fit one LabelEncoder per categorical column on the training data."""
        self.encoders = {}
        for col in CATEGORICAL_COLS:
            enc = LabelEncoder()
            # Ensure consistent string typing before fitting
            enc.fit(df[col].astype(str).values)
            self.encoders[col] = enc
        self.fitted = True

    def apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted encoders to a DataFrame. Unknown labels at inference time
        (categories not seen during training) are mapped to a value
        (-1) rather than raising - this allows the prototype to gracefully
        handle previously-unseen departments or countries.
        """
        if not self.fitted:
            raise RuntimeError(
                "Preprocessor encoders are not fitted. Call fit_encoders first "
                "or load a fitted preprocessor from disk via Preprocessor.load()."
            )

        df = df.copy()
        for col in CATEGORICAL_COLS:
            enc = self.encoders[col]
            # Build a set for O(1) lookup of "have we seen this label?"
            known = set(enc.classes_)
            values = df[col].astype(str).values
            # Map unknown values to -1 sentinel; known values to the
            # corresponding encoded integer. The -1 lets tree-based models
            # split on "previously unseen category" without raising at inference.
            df[col] = [
                enc.transform([v])[0] if v in known else -1 for v in values
            ]
        return df

    # Full pipeline

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit encoders and produce the model-ready feature matrix + target.

        Used during training. Returns (X, y) where X has the original numeric
        columns, encoded categorical columns, and engineered features.
        """
        df = self.engineer_features(df)
        self.fit_encoders(df)
        df = self.apply_encoders(df)

        feature_cols = CATEGORICAL_COLS + NUMERIC_COLS + DERIVED_COLS
        X = df[feature_cols]
        y = df[TARGET_COL]
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform records using already-fitted encoders. Used at inference.
        Does NOT require the target column to be present.
        """
        df = self.engineer_features(df)
        df = self.apply_encoders(df)
        feature_cols = CATEGORICAL_COLS + NUMERIC_COLS + DERIVED_COLS
        return df[feature_cols]

    # Persistence

    def save(self, path: str) -> None:
        """Persist the fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        """Load a previously-fitted preprocessor from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a Preprocessor: {type(obj)}")
        return obj

    # Convenience

    @staticmethod
    def feature_names() -> list[str]:
        """Return the ordered list of feature column names produced."""
        return CATEGORICAL_COLS + NUMERIC_COLS + DERIVED_COLS
