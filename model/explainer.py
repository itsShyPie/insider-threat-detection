"""
Explainability engine for the Insider Threat Detection System.

Produces three layers of explanation, in increasing accessibility:

  1. Per-prediction feature importance - which features drove the decision
  2. Categorised risk score breakdown - risk grouped into 5 domains
  3. Plain-English explanation text - for non-technical security managers

The plain-English layer is the binding requirement (FR-15, FR-16) that
distinguishes this system from a black-box classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .predict import PredictionResult
from .preprocessor import (
    CATEGORICAL_COLS,
    DERIVED_COLS,
    NUMERIC_COLS,
    Preprocessor,
)


# Training CSV used to compute the per-feature statistics that drive the
# per-record attribution. Same file used by train.py.
TRAINING_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "insider_threat_clean_dataset.csv"
)


# Risk category mapping
# Each behavioural feature is mapped to one of five risk categories. This
# grouping is what the manager-facing explanation summarises.

RISK_CATEGORIES: dict[str, list[str]] = {
    "Printing Behaviour": [
        "total_printed_pages",
        "num_printed_pages_off_hours",
        "off_hours_print_ratio",
    ],
    "Data Transfer Activity": [
        "total_files_burned",
        "burned_from_other",
        "files_burned_per_entry",
    ],
    "Access Patterns": [
        "num_entries",
        "num_unique_campus",
        "late_exit_flag",
        "entry_during_weekend",
    ],
    "Travel & Location Risk": [
        "is_abroad",
        "trip_day_number",
        "hostility_country_level",
    ],
    "User Profile": [
        "employee_seniority_years",
        "is_contractor",
        "employee_classification",
        "has_foreign_citizenship",
        "has_criminal_record",
        "has_medical_history",
        "employee_department",
        "employee_campus",
        "employee_position",
        "employee_origin_country",
    ],
}

# Anomaly rules (the plain-English layer)
# These rules are NOT used by the model for prediction - the Random Forest
# is the  classifier. The rules exist to translate the
# decision into manager-friendly language: when rule fires on the input
# record, the rule message is appended to the explanation text.
#
# Each entry is (rule_id, predicate, message_template) tuple. The predicate
# is lambda that takes the raw feature dict and returns True if the
# rule should fire. The message_template is Python format string;
# placeholders like {pct}, {n_files}, {n_campus} are filled in by
# _detect_anomalies() below.

ANOMALY_RULES: list[tuple[str, callable, str]] = [
    (
        "off_hours_printing",
        lambda f: f.get("off_hours_print_ratio", 0) > 0.5,
        "A high proportion of printing activity ({pct:.0%}) "
        "occurred outside normal business hours."
    ),
    (
        "removable_media_volume",
        lambda f: f.get("total_files_burned", 0) > 10,
        "{n_files:,.0f} files have been written to removable media - "
        "significantly above the typical employee average."
    ),
    (
        "burned_from_other",
        lambda f: f.get("burned_from_other", 0) >= 1,
        "Files belonging to other users were burned to removable media, "
        "which may indicate privilege misuse."
    ),
    (
        "weekend_access",
        lambda f: f.get("entry_during_weekend", 0) >= 1,
        "Premises access recorded during weekend hours."
    ),
    (
        "late_exit",
        lambda f: f.get("late_exit_flag", 0) >= 1,
        "Premises exit recorded significantly later than usual."
    ),
    (
        "multi_campus",
        lambda f: f.get("num_unique_campus", 0) >= 3,
        "Activity was recorded across {n_campus:.0f} different campuses, "
        "which is atypical for most roles."
    ),
    (
        "hostile_travel",
        lambda f: f.get("hostility_country_level", 0) >= 2,
        "A recent travel record involves a country flagged as elevated "
        "geopolitical risk."
    ),
    (
        "contractor_risk",
        lambda f: f.get("is_contractor", 0) == 1
                  and f.get("employee_classification", 0) >= 2,
        "Contractor status combined with an elevated access classification "
        "- a profile that warrants closer monitoring."
    ),
    (
        "criminal_record",
        lambda f: f.get("has_criminal_record", 0) == 1,
        "A prior criminal record is on file."
    ),
]


# Output dataclass

@dataclass
class Explanation:
    """Structured explanation payload returned to the UI."""
    classification: int
    label: str
    confidence: float
    feature_importance: dict[str, float]         # Top N features ranked
    risk_breakdown: dict[str, float]             # 5 risk categories, normalised
    anomalies: list[str]                         # Triggered anomaly messages
    text: str                                    # Plain-English explanation
    recommendation: str                          # Action suggestion


# Explainer

class Explainer:
    """
    Generates a layered explanation for a single prediction.

    Usage:
        explainer = Explainer(predictor)
        exp = explainer.explain(record_df)
        print(exp.text)
    """

    def __init__(self, predictor, top_n_features: int = 10) -> None:
        self.predictor = predictor
        self.top_n_features = top_n_features
        # Global Gini importance from the trained Random Forest. Used as the
        # weight source for per-record attribution and as the global view on
        # the Model Insights tab.
        self.global_importance = predictor.get_feature_importances()
        # Training-set statistics used to convert a single record feature
        # value into a deviation score (|z-score| for numerics, rarity for
        # categoricals). Per-record attribution is then global_importance
        # weighted by this deviation, so the Single Record charts reflect
        # this record rather than the model average behaviour (FR-13/14).
        self._feature_stats = self._compute_feature_stats()

    # Main entry point

    def explain(self, record: pd.DataFrame) -> Explanation:
        result = self.predictor.predict(record)

        # Always present raw values (pre-encoding) for the anomaly rules.
        # The rules look at the human-readable feature values, not encoded ones.
        raw_values = self._extract_raw_values(record)

        # Per-record attribution: weight the global importance of each
        # feature by how unusual the record value for that feature is.
        # This is what makes the Single Record charts genuinely per-prediction.
        attribution = self._per_record_attribution(raw_values)

        feature_importance = self._top_feature_importance(attribution)
        risk_breakdown = self._categorised_risk_breakdown(attribution)
        anomalies = self._detect_anomalies(raw_values)
        text = self._generate_text(result, anomalies, risk_breakdown)
        recommendation = self._generate_recommendation(result)

        return Explanation(
            classification=result.classification,
            label=result.label,
            confidence=result.confidence,
            feature_importance=feature_importance,
            risk_breakdown=risk_breakdown,
            anomalies=anomalies,
            text=text,
            recommendation=recommendation,
        )

    # Internals

    @staticmethod
    def _extract_raw_values(record: pd.DataFrame) -> dict[str, Any]:
        """Pull raw values from the un-preprocessed record + compute derived ones."""
        row = record.iloc[0].to_dict()
        # Compute derived features the way preprocessor.engineer_features does
        row["off_hours_print_ratio"] = (
            row.get("num_printed_pages_off_hours", 0)
            / (row.get("total_printed_pages", 0) + 1)
        )
        row["files_burned_per_entry"] = (
            row.get("total_files_burned", 0)
            / (row.get("num_entries", 0) + 1)
        )
        return row

    def _compute_feature_stats(self) -> dict:
        """
        Compute per-feature training-set statistics used by per-record
        attribution. Numeric/engineered features get (mean, std); categorical
        features get {value: frequency} map.

        Loaded once at Explainer construction. If the training CSV is not
        present, per-record attribution silently degrades to plain global
        importance, so the prototype still runs.
        """
        stats: dict[str, dict] = {"numeric": {}, "categorical": {}}
        if not TRAINING_DATA_PATH.exists():
            return stats

        df = pd.read_csv(TRAINING_DATA_PATH)
        # Engineered features (off_hours_print_ratio, files_burned_per_entry)
        # must be computed the same way the preprocessor computes them so the
        # record value and the training mean live on the same scale.
        df_eng = Preprocessor.engineer_features(df)

        for col in NUMERIC_COLS + DERIVED_COLS:
            if col in df_eng.columns:
                vals = df_eng[col].astype(float)
                mean = float(vals.mean())
                std = float(vals.std())
                # Guard against zero-variance columns producing division errors.
                stats["numeric"][col] = (mean, std if std > 0 else 1.0)

        for col in CATEGORICAL_COLS:
            if col in df.columns:
                freq = df[col].astype(str).value_counts(normalize=True).to_dict()
                stats["categorical"][col] = freq

        return stats

    def _per_record_attribution(self, raw_values: dict) -> dict[str, float]:
        """
        Per-record feature attribution: global_importance × deviation.

        - Numeric/engineered features: deviation = |z-score| against training
          mean/std. A z of 0 means "exactly average for the population" and
          contributes nothing to the attribution; a large |z| means
          the record value is unusual and so the feature matters more for
          this prediction.
        - Categorical features: deviation = 1 − training frequency, so a rare
          department or country contributes more than a common one. Unseen
          values get the maximum rarity of 1.

        The result is normalised to sum to 1 across all features.
        """
        attrs: dict[str, float] = {}

        for col, (mean, std) in self._feature_stats.get("numeric", {}).items():
            if col not in self.global_importance:
                continue
            v = float(raw_values.get(col, mean))
            z = abs((v - mean) / std)
            attrs[col] = self.global_importance[col] * z

        for col, freq in self._feature_stats.get("categorical", {}).items():
            if col not in self.global_importance:
                continue
            v = str(raw_values.get(col, ""))
            rarity = 1.0 - freq.get(v, 0.0)
            attrs[col] = self.global_importance[col] * rarity

        # Any model feature we have no stats for falls back to plain global
        # importance so it still appears in the ranking.
        for col, imp in self.global_importance.items():
            attrs.setdefault(col, imp)

        s = sum(attrs.values()) or 1.0
        return {k: v / s for k, v in attrs.items()}

    def _top_feature_importance(self, attribution: dict[str, float]) -> dict[str, float]:
        """Return the top N features by per-record attribution, normalised to sum to 1."""
        items = sorted(attribution.items(), key=lambda x: x[1], reverse=True)
        top = items[:self.top_n_features]
        total = sum(v for _, v in top) or 1.0
        return {k: v / total for k, v in top}

    def _categorised_risk_breakdown(self, attribution: dict[str, float]) -> dict[str, float]:
        """
        Aggregate per-record attribution by risk category, returning a
        normalised score per category that sums to 1.0 across all five.
        """
        scores = {}
        for category, features in RISK_CATEGORIES.items():
            scores[category] = sum(attribution.get(f, 0.0) for f in features)

        s = sum(scores.values()) or 1.0
        return {k: v / s for k, v in scores.items()}

    def _detect_anomalies(self, raw_values: dict) -> list[str]:
        """Run each anomaly rule, format any that fire."""
        messages = []
        for rule_id, predicate, template in ANOMALY_RULES:
            try:
                if predicate(raw_values):
                    msg = template.format(
                        pct=raw_values.get("off_hours_print_ratio", 0),
                        n_files=raw_values.get("total_files_burned", 0),
                        n_campus=raw_values.get("num_unique_campus", 0),
                    )
                    messages.append(msg)
            except Exception:
                # Defensive: don't let a rule crash the explainer
                continue
        return messages

    def _fired_rule_ids(self, raw_values: dict) -> list[str]:
        """Return the IDs of anomaly rules that fired for this record."""
        fired = []
        for rule_id, predicate, _ in ANOMALY_RULES:
            try:
                if predicate(raw_values):
                    fired.append(rule_id)
            except Exception:
                continue
        return fired

    # Batch helpers

    def per_record_summary(self, record: pd.DataFrame) -> dict:
        """
        Per-record info for use in batch tables. Returns the top
        contributing feature, the share of the per-record attribution, and
        the IDs of any anomaly rules that fired.

        Cheaper than `explain()` because it skips the plain-English text and
        recommendation steps. Does NOT call the predictor.
        """
        raw_values = self._extract_raw_values(record)
        attribution = self._per_record_attribution(raw_values)
        top_feature, top_weight = max(attribution.items(), key=lambda x: x[1])
        return {
            "top_feature": top_feature,
            "top_weight": float(top_weight),
            "fired_rule_ids": self._fired_rule_ids(raw_values),
        }

    @staticmethod
    def _generate_text(result: PredictionResult,
                       anomalies: list[str],
                       risk_breakdown: dict[str, float]) -> str:
        """Compose the full plain-English explanation."""
        pct = int(result.confidence * 100)

        if result.classification == 1:
            opener = (
                f"This record has been flagged as potentially malicious "
                f"with {pct}% confidence."
            )
        else:
            opener = (
                f"This record appears consistent with normal behaviour "
                f"({pct}% confidence)."
            )

        if anomalies:
            body = (
                " The classification was influenced by the following observed "
                "patterns:\n" + "\n".join(f"  • {a}" for a in anomalies)
            )
        else:
            body = (
                " No individual behavioural pattern stood out as significantly "
                "abnormal."
            )

        # Mention the top risk category by weight (per-record attribution).
        top_cat = max(risk_breakdown.items(), key=lambda x: x[1])
        footer = (
            f"\n\nThe risk category contributing most to this assessment "
            f"is '{top_cat[0]}' ({top_cat[1]:.0%} of the per-record attribution)."
        )

        return opener + body + footer

    @staticmethod
    def _generate_recommendation(result: PredictionResult) -> str:
        """Produce a manager-facing action recommendation."""
        if result.classification == 1:
            if result.confidence > 0.85:
                return (
                    "Recommendation: Escalate this case for review by the "
                    "security team. This is an automated assessment and must "
                    "be validated by a human investigator before any action "
                    "is taken."
                )
            return (
                "Recommendation: Mark this case for monitoring. Confidence "
                "is moderate; collect additional context before escalating."
            )
        return (
            "Recommendation: No action required at this time. Continue routine "
            "monitoring."
        )
