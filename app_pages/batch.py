"""Batch sentiment analysis."""

import altair as alt
import pandas as pd
import streamlit as st

import model_utils

SAMPLE_INPUT = (
    "I absolutely love this product. It works perfectly.\n"
    "This is terrible and I regret buying it.\n"
    "The product is okay, nothing special.\n"
    "I am not happy with the service.\n"
    "Outstanding quality and friendly customer care!\n"
    "Worst purchase ever made, total waste of money.\n"
    "Highly recommended for anyone looking for good value."
)

st.header("Batch analysis", icon=":material/dataset:", divider="blue")
st.caption(
    "Classify many sentences in a single Spark execution. Enter one sentence per line "
    "or upload a text file, then run the classification."
)

if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None

batch_text = st.text_area("Sentences, one per line", value=SAMPLE_INPUT, height=180)

uploaded_file = st.file_uploader(
    "Or upload a text file (.txt, one sentence per line)", type=["txt"]
)
if uploaded_file is not None:
    try:
        batch_text = uploaded_file.getvalue().decode("utf-8")
        st.caption(
            f"Loaded {len(batch_text.splitlines())} lines from {uploaded_file.name}."
        )
    except Exception as exc:
        st.error(
            f"Could not read the uploaded file: {exc}",
            icon=":material/error:",
        )

if st.button("Run classification", type="primary", icon=":material/play_arrow:"):
    lines = [line.strip() for line in batch_text.splitlines() if line.strip()]
    if not lines:
        st.warning(
            "Provide at least one sentence to classify.",
            icon=":material/warning:",
        )
    else:
        try:
            with st.spinner(
                f"Classifying {len(lines)} sentences through the Spark pipeline..."
            ):
                results = model_utils.predict_batch_sentiment(lines)
            st.session_state["batch_results"] = results
        except Exception as exc:
            st.session_state["batch_results"] = None
            st.error(
                f"Batch classification failed: {exc}",
                title="Pipeline error",
                icon=":material/error:",
            )

results = st.session_state["batch_results"]
if results is not None and not results.empty:
    total = len(results)
    positive = int((results["Predicted Sentiment"] == "Positive").sum())
    negative = total - positive
    dominant = "Positive" if positive >= negative else "Negative"

    with st.container(horizontal=True):
        st.metric("Sentences classified", total, border=True)
        st.metric("Positive", positive, border=True)
        st.metric("Negative", negative, border=True)
        st.metric("Dominant sentiment", dominant, border=True)

    dark_mode = st.context.theme.type == "dark"
    positive_color = "#34D399" if dark_mode else "#10B981"
    negative_color = "#F87171" if dark_mode else "#EF4444"

    chart_df = pd.DataFrame(
        {"Sentiment": ["Positive", "Negative"], "Sentences": [positive, negative]}
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Sentiment:N", title=None, sort=["Positive", "Negative"]),
            y=alt.Y("Sentences:Q", title="Sentences"),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["Positive", "Negative"],
                    range=[positive_color, negative_color],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Sentiment:N", title="Sentiment"),
                alt.Tooltip("Sentences:Q", title="Sentences"),
            ],
        )
    )
    with st.container(border=True):
        st.subheader("Sentiment distribution")
        st.altair_chart(
            chart,
            alt="Bar chart of positive and negative sentence counts",
        )

    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Detailed predictions")
        st.badge(f"Positive: {positive}", color="green")
        st.badge(f"Negative: {negative}", color="red")

    st.dataframe(
        results,
        hide_index=True,
        column_order=[
            "Index",
            "Original Sentence",
            "Cleaned Sentence",
            "Predicted Sentiment",
            "Positive Probability (%)",
            "Negative Probability (%)",
        ],
        column_config={
            "Index": st.column_config.NumberColumn("No.", format="%d"),
            "Original Sentence": st.column_config.TextColumn(
                "Original sentence", width="large"
            ),
            "Cleaned Sentence": st.column_config.TextColumn(
                "Preprocessed text", width="medium"
            ),
            "Predicted Sentiment": st.column_config.TextColumn("Prediction"),
            "Positive Probability (%)": st.column_config.ProgressColumn(
                "Positive probability (%)", min_value=0, max_value=100, format="%.2f"
            ),
            "Negative Probability (%)": st.column_config.ProgressColumn(
                "Negative probability (%)", min_value=0, max_value=100, format="%.2f"
            ),
        },
        alt="Table of sentences with predicted sentiment and class probabilities",
    )

    st.download_button(
        "Download results as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="sentimentscope_batch_results.csv",
        mime="text/csv",
        icon=":material/download:",
    )
