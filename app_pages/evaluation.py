"""Model evaluation dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_FILE = Path(__file__).resolve().parent.parent / "sentiment_evaluation_results.json"


@st.cache_data
def load_evaluation_data(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


st.header("Model evaluation", icon=":material/query_stats:", divider="green")
st.caption(
    "Empirical performance of the fitted pipeline on the held-out 30 percent test split."
)

eval_data = load_evaluation_data(str(RESULTS_FILE))

if eval_data is None:
    st.info(
        "The file `sentiment_evaluation_results.json` was not found or could not be "
        "parsed. Model performance metrics are unavailable.",
        title="Evaluation data unavailable",
        icon=":material/info:",
    )
else:
    metrics = eval_data.get("metrics", {})
    split = eval_data.get("split", {})
    pos_metrics = metrics.get("positive", {})
    neg_metrics = metrics.get("negative", {})
    cm = eval_data.get("confusion_matrix", {})

    with st.container(horizontal=True):
        st.metric(
            "Test accuracy",
            f"{metrics.get('accuracy', 0) * 100:.2f}%",
            border=True,
            icon=":material/verified:",
        )
        st.metric(
            "ROC-AUC",
            f"{metrics.get('roc_auc', 0):.4f}",
            border=True,
            icon=":material/monitoring:",
        )
        st.metric(
            "Training samples",
            f"{split.get('train_samples', 0):,}",
            border=True,
            icon=":material/database:",
        )
        st.metric(
            "Test samples",
            f"{split.get('test_samples', 0):,}",
            border=True,
            icon=":material/table_rows:",
        )

    metrics_col, matrix_col = st.columns(2)

    with metrics_col:
        with st.container(border=True, height="stretch"):
            st.subheader("Class-specific metrics")
            perf_df = pd.DataFrame(
                {
                    "Positive class (1)": [
                        pos_metrics.get("precision", 0),
                        pos_metrics.get("recall", 0),
                        pos_metrics.get("f1_score", 0),
                    ],
                    "Negative class (0)": [
                        neg_metrics.get("precision", 0),
                        neg_metrics.get("recall", 0),
                        neg_metrics.get("f1_score", 0),
                    ],
                },
                index=["Precision", "Recall", "F1-score"],
            )
            def green_intensity(column):
                low, high = column.min(), column.max()
                span = (high - low) or 1
                return [
                    "background-color: rgba(16, 185, 129, "
                    f"{0.08 + 0.35 * ((value - low) / span):.3f})"
                    for value in column
                ]

            st.dataframe(
                perf_df.style.apply(green_intensity).format("{:.4f}"),
                alt="Table of precision, recall, and F1-score per sentiment class",
            )

    with matrix_col:
        with st.container(border=True, height="stretch"):
            st.subheader("Confusion matrix (test split)")
            matrix_df = pd.DataFrame(
                [
                    [
                        cm.get("true_negative", 0),
                        cm.get("false_positive", 0),
                    ],
                    [
                        cm.get("false_negative", 0),
                        cm.get("true_positive", 0),
                    ],
                ],
                index=["Actual negative", "Actual positive"],
                columns=["Predicted negative", "Predicted positive"],
            )

            def highlight_intensity(column):
                peak = matrix_df.to_numpy().max() or 1
                return [
                    f"background-color: rgba(79, 70, 229, {0.08 + 0.38 * (value / peak):.3f})"
                    for value in column
                ]

            st.dataframe(
                matrix_df.style.apply(highlight_intensity).format("{:,.0f}"),
                alt="Confusion matrix of actual versus predicted class counts",
            )
            st.caption("Rows are actual labels; columns are predicted labels.")

    st.caption(
        "Class metrics are computed on the full held-out test split (30 percent of "
        "records). The stop-word remover preserves sentiment-critical negations such "
        "as no, not, and nor. The fitted vectorizer and regression coefficients are "
        "frozen, and the app never refits the model."
    )
