"""Single sentence sentiment analysis."""

import streamlit as st

import model_utils

EXAMPLES = [
    "I absolutely love this product. It works perfectly.",
    "This is terrible and I regret buying it.",
    "The product is okay, nothing special.",
    "I am not happy with the service.",
]

st.header("Single sentence analysis", icon=":material/search:", divider="violet")
st.caption(
    "Classify one sentence with the saved pipeline. The input is preprocessed exactly "
    "as during training before scoring."
)

if "analyzer_text" not in st.session_state:
    st.session_state["analyzer_text"] = ""
if "analyzer_result" not in st.session_state:
    st.session_state["analyzer_result"] = None


def apply_example():
    value = st.session_state.get("analyzer_example")
    if value:
        st.session_state["analyzer_text"] = value
        st.session_state["analyzer_result"] = None


def clear_input():
    st.session_state["analyzer_text"] = ""
    st.session_state["analyzer_example"] = None
    st.session_state["analyzer_result"] = None


st.pills(
    "Examples",
    EXAMPLES,
    key="analyzer_example",
    on_change=apply_example,
    help="Select an example to load it into the input below.",
)

st.text_area(
    "Sentence",
    key="analyzer_text",
    height=120,
    placeholder="Type or paste a sentence to analyze...",
)

text_value = st.session_state["analyzer_text"]
char_count = len(text_value)
word_count = len(text_value.split()) if text_value.strip() else 0
st.caption(f"{char_count} characters | {word_count} words")

with st.container(horizontal=True):
    analyze_clicked = st.button(
        "Analyze sentiment",
        type="primary",
        icon=":material/search:",
    )
    st.button("Clear", icon=":material/delete:", on_click=clear_input)

if analyze_clicked:
    stripped = st.session_state["analyzer_text"].strip()
    if not stripped:
        st.warning(
            "Enter at least one word before analyzing.",
            icon=":material/warning:",
        )
    else:
        try:
            with st.spinner("Scoring with the PySpark pipeline..."):
                result = model_utils.predict_single_sentiment(stripped)
            st.session_state["analyzer_result"] = result
        except Exception as exc:
            st.session_state["analyzer_result"] = None
            st.error(
                f"Scoring failed: {exc}",
                title="Pipeline error",
                icon=":material/error:",
            )

result = st.session_state["analyzer_result"]
if result is not None:
    if result.get("is_empty"):
        st.warning(result.get("error", "Input is empty."), icon=":material/warning:")
    else:
        with st.container(border=True):
            if result.get("zero_tokens"):
                st.info(
                    result["explanation"],
                    title="No usable tokens",
                    icon=":material/info:",
                )
            else:
                sentiment = result["sentiment"]
                is_positive = sentiment == "Positive"
                pos_pct = result["positive_percentage"]
                neg_pct = result["negative_percentage"]

                if is_positive:
                    st.success(
                        f"Predicted sentiment: **{sentiment}** "
                        f"({pos_pct:.2f}% positive probability)",
                        icon=":material/thumb_up:",
                    )
                else:
                    st.error(
                        f"Predicted sentiment: **{sentiment}** "
                        f"({neg_pct:.2f}% negative probability)",
                        icon=":material/thumb_down:",
                    )

                prob_positive, prob_negative = st.columns(2)
                prob_positive.metric(
                    "Positive probability", f"{pos_pct:.2f}%", border=True
                )
                prob_negative.metric(
                    "Negative probability", f"{neg_pct:.2f}%", border=True
                )

                st.caption(result["explanation"])

                st.markdown("**Preprocessed input sent to Spark**")
                st.code(result["cleaned_text"], language=None)

                st.caption(
                    "Probabilities are model-estimated outputs of logistic regression "
                    "and do not represent guaranteed confidence."
                )
