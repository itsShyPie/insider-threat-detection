"""
Streamlit prototype for the Insider Threat Detection System.

Two tabs:
  1. Analysis        - upload one or many records; routes to a single-record
                       explanation or a batch dashboard with click-to-inspect.
  2. Model & About   - global Random Forest behaviour, plus system framing,
                       ethics, and limitations.

Includes:
  - Manager-friendly results with severity tier
  - Risk score breakdown across five behavioural categories
  - Top-N feature importance chart
  - Plain-English explanation text
  - Specific recommended actions
  - Batch mode with sortable results table and aggregate dashboard

Run:
    streamlit run app.py
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.explainer import Explainer
from model.predict import Predictor
from model.preprocessor import CATEGORICAL_COLS, NUMERIC_COLS

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_COLUMNS = CATEGORICAL_COLS + NUMERIC_COLS

# Page configuration

st.set_page_config(
    page_title="Insider Threat Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .severity-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0 1rem 0;
        color: white;
    }
    .severity-critical { background: linear-gradient(135deg,#8B0000,#C00000); }
    .severity-high     { background: linear-gradient(135deg,#C04A1B,#E67817); }
    .severity-medium   { background: linear-gradient(135deg,#B17B00,#D89E2C); }
    .severity-low      { background: linear-gradient(135deg,#1F3864,#2E75B6); }
    .severity-clean    { background: linear-gradient(135deg,#385723,#548235); }
    .severity-card h2  { color: white !important; margin: 0; font-size: 1.4rem; }
    .severity-card p   { color: rgba(255,255,255,0.92) !important; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .footer-note { color:#888; font-size:0.85rem; margin-top:2rem; border-top:1px solid #eee; padding-top:0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Resource loading

@st.cache_resource(show_spinner=False)
def load_predictor_and_explainer():
    predictor = Predictor()
    explainer = Explainer(predictor, top_n_features=10)
    return predictor, explainer


# Severity tier

def severity_tier(prediction: int, confidence: float) -> tuple[str, str, str]:
    """Returns (tier_name, css_class, description)."""
    if prediction == 0:
        return ("CLEAR", "severity-clean",
                "No threat indicators identified. Routine monitoring is sufficient.")
    if confidence >= 0.85:
        return ("CRITICAL", "severity-critical",
                "Immediate review by the security team is strongly recommended.")
    if confidence >= 0.70:
        return ("HIGH", "severity-high",
                "Strong indicators of suspicious behaviour. Escalate for human review.")
    if confidence >= 0.55:
        return ("MEDIUM", "severity-medium",
                "Moderate confidence. Monitor closely and gather additional context.")
    return ("LOW", "severity-low",
            "Weak threat signal detected. Continue observation; no immediate action.")


# Validation

def validate_record(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )
    if len(df) == 0:
        errors.append("The uploaded file contains no data rows.")
    return (len(errors) == 0, errors)


# Rendering helpers

def render_severity_card(tier: str, css_class: str, description: str,
                          confidence: float) -> None:
    st.markdown(
        f"""
        <div class="severity-card {css_class}">
            <h2>{tier} &nbsp;·&nbsp; {confidence:.0%} confidence</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_breakdown_chart(
    risk_scores: dict[str, float],
    title: str = "Risk Category Breakdown",
    xaxis_title: str = "Share of model attention",
) -> None:
    df = pd.DataFrame(
        sorted(risk_scores.items(), key=lambda kv: kv[1]),
        columns=["Category", "Weight"],
    )
    fig = go.Figure(go.Bar(
        x=df["Weight"],
        y=df["Category"],
        orientation="h",
        marker=dict(
            color=df["Weight"],
            colorscale=[[0, "#DEEAF1"], [1, "#1F3864"]],
            line=dict(color="#1F3864", width=1),
        ),
        text=[f"{v:.0%}" for v in df["Weight"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=None,
        height=300,
        margin=dict(l=20, r=80, t=40, b=20),
        showlegend=False,
        xaxis=dict(tickformat=".0%", range=[0, max(df["Weight"]) * 1.25]),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance_chart(
    importance: dict[str, float],
    title: str = "Top Features by Model Importance",
    xaxis_title: str = "Relative importance",
) -> None:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    df = pd.DataFrame(items, columns=["Feature", "Importance"])
    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(color="#2E75B6", line=dict(color="#1F3864", width=1)),
        text=[f"{v:.1%}" for v in df["Importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=None,
        height=380,
        margin=dict(l=20, r=80, t=40, b=20),
        showlegend=False,
        xaxis=dict(tickformat=".0%", range=[0, max(df["Importance"]) * 1.25]),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_explanation(explanation, record: pd.DataFrame) -> None:
    tier, css_class, desc = severity_tier(explanation.classification, explanation.confidence)
    render_severity_card(tier, css_class, desc, explanation.confidence)

    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.markdown("### Plain-English Explanation")
        for line in explanation.text.split("\n"):
            if line.strip():
                st.markdown(line)

        st.markdown("### Recommended Action")
        st.info(explanation.recommendation)

        if explanation.anomalies:
            st.markdown("### Specific Behavioural Anomalies")
            for a in explanation.anomalies:
                st.markdown(f"- {a}")

    with col_right:
        render_risk_breakdown_chart(
            explanation.risk_breakdown,
            title="Risk Categories Driving This Prediction",
            xaxis_title="Per-record attribution",
        )
        render_feature_importance_chart(
            explanation.feature_importance,
            title="Top Features Driving This Prediction",
            xaxis_title="Per-record attribution",
        )

    with st.expander("View raw input record"):
        st.dataframe(record.T.rename(columns={record.index[0]: "Value"}),
                     use_container_width=True)


# Batch analysis

@st.cache_data(show_spinner=False)
def run_batch_analysis(file_bytes: bytes, _predictor, _explainer) -> dict:
    """Parse the uploaded CSV, classify every record, compute per-record
    attribution + fired anomaly rules, and assemble the summary DataFrame.

    The leading-underscore args are excluded from the cache key by
    Streamlit convention - they stay constant for the
    lifetime of the Streamlit process.
    """
    df_in = pd.read_csv(BytesIO(file_bytes))

    id_cols = [
        c for c in ["profile_name", "employee_id", "user_id", "name"]
        if c in df_in.columns
    ]
    target_col = "is_malicious" if "is_malicious" in df_in.columns else None
    feature_df = df_in.drop(
        columns=id_cols + ([target_col] if target_col else []),
        errors="ignore",
    ).reset_index(drop=True)

    ok, errors = validate_record(feature_df)
    if not ok:
        return {"ok": False, "errors": errors}

    results = _predictor.predict_batch(feature_df)
    per_record = [
        _explainer.per_record_summary(feature_df.iloc[[i]])
        for i in range(len(feature_df))
    ]

    summary = pd.DataFrame({
        "Row": range(len(results)),
        "Prediction": [r.label for r in results],
        "Severity": [severity_tier(r.classification, r.confidence)[0] for r in results],
        "P(Malicious)": [r.probability_malicious for r in results],
        "Confidence": [r.confidence for r in results],
        "Top Driver": [pr["top_feature"] for pr in per_record],
        "Driver Weight": [pr["top_weight"] for pr in per_record],
        "Anomalies": [", ".join(pr["fired_rule_ids"]) or "-" for pr in per_record],
    })
    # Pull a few raw behavioural values into the table so the
    # investigator can see at a glance *why* a row was flagged
    # without having to drill in.
    for raw_col, display_name in [
        ("total_files_burned", "Files Burned"),
        ("total_printed_pages", "Pages Printed"),
        ("num_unique_campus", "Campuses"),
    ]:
        if raw_col in feature_df.columns:
            summary[display_name] = feature_df[raw_col].values

    for c in id_cols:
        summary.insert(1, c, df_in[c].values)
    if target_col:
        summary["Actual"] = (
            df_in[target_col].map({0: "Normal", 1: "Malicious"}).values
        )

    # Pre-compute everything else the dashboard needs from the heavy data
    # so we don't have to retain the large results list in the cache.
    rule_counter: Counter[str] = Counter()
    for pr in per_record:
        rule_counter.update(pr["fired_rule_ids"])

    return {
        "ok": True,
        "df_in": df_in,
        "feature_df": feature_df,
        "summary": summary,
        "rule_counts": dict(rule_counter),
        "y_pred": np.array([r.classification for r in results]),
        "p_malicious": np.array([r.probability_malicious for r in results]),
        "threshold": float(results[0].threshold) if results else float(_predictor.threshold),
        "id_cols": id_cols,
        "target_col": target_col,
    }


# Application body

st.title("Insider Threat Detection System")
st.caption("COS720  ·  University of Pretoria")

try:
    predictor, explainer = load_predictor_and_explainer()
except FileNotFoundError as e:
    st.error(
        "**Trained model not found.**\n\n"
        "Run `python -m model.train` from the project root, then refresh this page.\n\n"
        f"Details: {e}"
    )
    st.stop()

# Stale session-state keys from earlier layouts. Pop them so we don't
# carry forward unused widget state across UI changes.
for _legacy_key in (
    "batch_selected_record",
    "_pending_batch_focus",
    "single_source_method",
    "batch_file_id",
    "analysis_source",
):
    st.session_state.pop(_legacy_key, None)

tab_analysis, tab_model = st.tabs(["Analysis", "Model & About"])

# TAB 1 - Analysis
# The view auto-routes by record count: 1-row CSV uploads render the
# per-record explanation directly; multi-row CSV uploads render the
# batch dashboard, and clicking any row drills in inline below the tables.

with tab_analysis:
    st.markdown("### Input")

    record_df: pd.DataFrame | None = None
    batch_payload: dict | None = None

    uploaded = st.file_uploader(
        "Upload a CSV file - one record or many",
        type=["csv"],
        key="analysis_upload",
    )
    if uploaded is not None:
        try:
            file_bytes = uploaded.getvalue()
            df_in_peek = pd.read_csv(BytesIO(file_bytes))
        except Exception as e:
            st.error(
                f"Could not parse the uploaded file. Please upload a valid CSV.\n\nDetails: {e}"
            )
            file_bytes = None
            df_in_peek = None

        if df_in_peek is not None:
            if len(df_in_peek) == 0:
                st.error("The uploaded file contains no data rows.")
            elif len(df_in_peek) == 1:
                # 1-row CSV: treat as a single-record analysis. Strip
                # the same non-feature columns the cached batch path
                # would strip, then validate.
                single = df_in_peek.drop(
                    columns=[c for c in ("is_malicious", "profile_name")
                             if c in df_in_peek.columns],
                    errors="ignore",
                ).reset_index(drop=True)
                ok, errors = validate_record(single)
                if not ok:
                    for msg in errors:
                        st.error(msg)
                else:
                    record_df = single
            else:
                # Multi-row CSV: batch dashboard. First call is slow,
                # subsequent reruns hit the cache and return instantly.
                try:
                    with st.spinner("Analysing batch..."):
                        batch_payload = run_batch_analysis(
                            file_bytes, predictor, explainer
                        )
                except Exception as e:
                    st.error(f"Could not analyse batch. Details: {e}")

    # Single-record render
    if record_df is not None:
        st.markdown("---")
        with st.spinner("Analysing behavioural record..."):
            explanation = explainer.explain(record_df)
        render_explanation(explanation, record_df)

    # Batch render
    elif batch_payload is not None:
        if not batch_payload["ok"]:
            for msg in batch_payload["errors"]:
                st.error(msg)
        else:
            df_in = batch_payload["df_in"]
            feature_df = batch_payload["feature_df"]
            summary = batch_payload["summary"]
            rule_counts = batch_payload["rule_counts"]
            y_pred = batch_payload["y_pred"]
            p_malicious = batch_payload["p_malicious"]
            threshold = batch_payload["threshold"]
            id_cols = batch_payload["id_cols"]
            target_col = batch_payload["target_col"]

            st.markdown("---")
            st.markdown("### Summary Dashboard")

            col1, col2, col3, col4 = st.columns(4)
            n_total = len(summary)
            n_mal = int((summary["Prediction"] == "Malicious").sum())
            pct_mal = n_mal / n_total if n_total else 0
            avg_conf = summary["Confidence"].mean()

            col1.metric("Records analysed", f"{n_total:,}")
            col2.metric("Flagged as malicious", f"{n_mal:,}", f"{pct_mal:.1%} of total")
            col3.metric("Normal", f"{n_total - n_mal:,}", f"{1 - pct_mal:.1%} of total")
            col4.metric("Avg confidence", f"{avg_conf:.0%}")

            # Ground-truth comparison (only when the CSV provides actuals).
            if target_col:
                from sklearn.metrics import (
                    confusion_matrix, precision_score, recall_score, f1_score,
                )
                y_true = df_in[target_col].astype(int).values
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn_, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                st.markdown("#### Ground-Truth Comparison")
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("True Positives", f"{tp:,}")
                g2.metric("False Positives", f"{fp:,}")
                g3.metric("False Negatives", f"{fn_:,}")
                g4.metric("True Negatives", f"{tn:,}")
                h1, h2, h3 = st.columns(3)
                h1.metric("Precision (Malicious)", f"{precision:.1%}")
                h2.metric("Recall (Malicious)", f"{recall:.1%}")
                h3.metric("F1 (Malicious)", f"{f1:.3f}")
                st.caption(
                    f"Operating threshold P(Malicious) ≥ {threshold:.2f}. "
                    "Recall-priority tuning trades some precision for fewer missed threats."
                )

            col_left, col_right = st.columns([1.0, 1.0])

            with col_left:
                sev_counts = summary["Severity"].value_counts().reindex(
                    ["CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAR"], fill_value=0
                )
                fig = go.Figure(go.Bar(
                    x=sev_counts.index,
                    y=sev_counts.values,
                    marker=dict(color=["#8B0000", "#C04A1B", "#B17B00", "#1F3864", "#548235"]),
                    text=sev_counts.values,
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Severity Tier Distribution",
                    xaxis_title=None, yaxis_title="Number of records",
                    height=320, margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                fig = go.Figure(go.Histogram(
                    x=summary["P(Malicious)"],
                    nbinsx=20,
                    marker=dict(color="#2E75B6", line=dict(color="#1F3864", width=1)),
                ))
                fig.update_layout(
                    title="Distribution of Malicious Probabilities",
                    xaxis_title="P(Malicious)", yaxis_title="Number of records",
                    height=320, margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(tickformat=".0%"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Anomaly rules fired across the batch - shows what behaviour
            # drove the flagged records, not just how many were flagged.
            if rule_counts:
                st.markdown("### Anomaly Rules Fired Across Batch")
                rule_df = pd.DataFrame(
                    sorted(rule_counts.items(), key=lambda kv: kv[1]),
                    columns=["Rule", "Records"],
                )
                fig = go.Figure(go.Bar(
                    x=rule_df["Records"],
                    y=rule_df["Rule"],
                    orientation="h",
                    marker=dict(color="#C04A1B", line=dict(color="#8B0000", width=1)),
                    text=rule_df["Records"],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=None,
                    xaxis_title="Records that fired the rule",
                    yaxis_title=None,
                    height=max(260, 30 * len(rule_df) + 80),
                    margin=dict(l=20, r=60, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Anomaly rules are descriptive (not used by the classifier). "
                    "They translate behavioural patterns into plain-English flags."
                )

            def _format_for_display(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out["Confidence"] = out["Confidence"].map(lambda x: f"{x:.0%}")
                out["P(Malicious)"] = out["P(Malicious)"].map(lambda x: f"{x:.1%}")
                out["Driver Weight"] = out["Driver Weight"].map(lambda x: f"{x:.0%}")
                return out

            st.markdown("### Highest-Risk Records")
            st.caption("Top 10 records by P(Malicious). Click any row to view the full per-record explanation below.")
            top_n = min(10, len(summary))
            top = summary.sort_values("P(Malicious)", ascending=False).head(top_n)
            top_event = st.dataframe(
                _format_for_display(top),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="batch_top_table",
            )

            st.markdown("### Full Results")
            st.caption("Click any row to view the full per-record explanation below.")
            full_event = st.dataframe(
                _format_for_display(summary),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="batch_full_table",
            )

            st.download_button(
                "Download results as CSV",
                summary.to_csv(index=False),
                "batch_results.csv",
                "text/csv",
            )

            # Inline drill-down - clicking a row renders the full
            # per-record explanation here
        
            def _row_from_event(event, source_df: pd.DataFrame) -> int | None:
                if not event.selection.rows:
                    return None
                pos = event.selection.rows[0]
                return int(source_df["Row"].iloc[pos])

            # Full Results wins if both tables have a selection. Explicit
            # None check because row index 0 is falsy.
            selected_idx = _row_from_event(full_event, summary)
            if selected_idx is None:
                selected_idx = _row_from_event(top_event, top)

            if selected_idx is not None:
                st.markdown("---")
                label_parts: list[str] = []
                if id_cols:
                    label_parts.append(str(df_in.iloc[selected_idx][id_cols[0]]))
                label_parts.append(f"Row {selected_idx}")
                label_parts.append(summary["Prediction"].iloc[selected_idx])
                label_parts.append(f"P(Mal)={p_malicious[selected_idx]:.0%}")
                label_parts.append(summary["Severity"].iloc[selected_idx])
                st.markdown(f"### Inspecting: {'  ·  '.join(label_parts)}")

                drill_record = feature_df.iloc[[selected_idx]].reset_index(drop=True)
                with st.spinner("Generating explanation..."):
                    drill_exp = explainer.explain(drill_record)
                render_explanation(drill_exp, drill_record)

# TAB 2 - Model & About

with tab_model:
    st.markdown("### Global Model Behaviour")
    st.caption(
        "These charts describe what the trained Random Forest has learned across "
        "the entire training set - independent of any single record."
    )

    importances = predictor.get_feature_importances()
    top15 = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:15])

    col1, col2 = st.columns([1.4, 1.0])

    with col1:
        render_feature_importance_chart(top15)

    with col2:
        st.markdown("#### Model Configuration")
        st.markdown(
            """
            - **Algorithm:** Random Forest Classifier
            - **Trees:** 200
            - **Max depth:** 20
            - **Class weight:** balanced (addresses base-rate fallacy)
            - **Features:** 23 (21 raw + 2 engineered)
            - **Training records:** 94,891
            - **Test records:** 23,723
            - **Held-out malicious:** 1,277
            """
        )

        st.markdown("#### Headline Performance")
        st.markdown(
            """
            | Metric | Value |
            |---|---|
            | Accuracy | 96.96% |
            | Recall (malicious) | 82.38% |
            | Precision (malicious) | 67.91% |
            | F1 (malicious) | 0.7445 |
            """
        )

    st.markdown("---")
    st.markdown("### About This System")
    st.markdown(
        """
        This prototype implements a **Host-Based Intrusion Detection System (HIDS)**
        for insider threat detection.
        """
    )

    st.markdown("#### How It Works")
    st.markdown(
        """
        1. **Input** - a behavioural record describing employee activity across
           printing, file transfer, physical access, travel, and user profile dimensions.
        2. **Preprocessing** - categorical features are label-encoded and two derived
           features are computed: `off_hours_print_ratio` and `files_burned_per_entry`.
        3. **Classification** - a Random Forest classifier with 200 trees, trained on
           94,891 records with class-weight balancing, produces a malicious/normal
           prediction and a probability.
        4. **Explainability** - the system surfaces top-ranked features, a five-category
           risk breakdown, and a plain-English explanation specifically formatted for
           non-technical security managers.
        5. **Recommendation** - based on prediction and confidence, the system suggests
           an action (clear / monitor / escalate / immediate review).
        """
    )

    st.markdown("#### Ethical Considerations")
    st.markdown(
        """
        The system is intended as a decision-support tool, not as an automated
        enforcement mechanism. Every flagged case must be reviewed by a human
        investigator before any personnel action is taken. The system operates within
        the constraints of the South African POPIA Act and similar privacy frameworks,
        which require transparent monitoring practices and proportionality between
        risk and surveillance.

        False positives carry real cost: a normal employee wrongly flagged faces HR
        investigation, damaged trust, and potential legal exposure. Threshold tuning
        prioritises catching genuine threats over minimising false alarms, on the
        basis that the cost of a missed insider exceeds the cost of a manual review,
        but this asymmetry must be communicated to operators.
        """
    )

    st.markdown("#### Limitations")
    st.markdown(
        """
        - The model is trained on a single benchmark dataset; real-world deployment
          would require retraining on organisation-specific data.
        - Per-record classification does not capture gradual behavioural drift over
          time; a sliding-window extension is a planned improvement.
        - Some features (e.g. `has_criminal_record`, `has_foreign_citizenship`) may
          introduce discriminatory bias if used without human oversight.
        - The system explains *what* features drove a decision, not *why* those
          features correlate with malicious activity - causal interpretation requires
          domain expertise.
        """
    )

    st.markdown(
        """
        <div class="footer-note">
        COS720 &nbsp;·&nbsp; University of Pretoria
        </div>
        """,
        unsafe_allow_html=True,
    )
