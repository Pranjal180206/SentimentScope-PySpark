"""
SentimentScope - PySpark sentiment analysis web application.

Entrypoint: page configuration, sidebar, shared header, and navigation.
Individual views live in app_pages/.
"""

import streamlit as st

import model_utils

st.set_page_config(
    page_title="SentimentScope",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def initialize_model():
    """Load the fitted PySpark pipeline once per server process."""
    return model_utils.get_model_health_status()


model_status = initialize_model()
engine_ready = model_status["status"] == "Ready"

# ----------------------------------------------------------------
# Sidebar: brand, engine status, project metadata
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### SentimentScope")
    st.caption("Big data sentiment analytics with PySpark")

    st.markdown("**Engine status**")
    if engine_ready:
        st.badge(
            "Pipeline loaded and active",
            icon=":material/check_circle:",
            color="green",
        )
        st.caption(
            f"PySpark {model_status.get('spark_version', 'N/A')} | Pre-trained PipelineModel"
        )
    else:
        st.badge("Engine offline", icon=":material/error:", color="red")
        st.caption(
            str(model_status.get("error_message", "Unknown initialization error"))
        )

    with st.expander("Diagnostics", icon=":material/build:"):
        st.write("**Java runtime:**", model_status.get("java_home", "N/A"))
        st.write("**Spark version:**", model_status.get("spark_version", "N/A"))
        st.write("**Model path:**", model_status.get("model_path", "N/A"))
        if "pipeline_stages" in model_status:
            st.write("**Pipeline stages:**")
            for i, stage in enumerate(model_status["pipeline_stages"], start=1):
                st.code(f"{i}. {stage}", language=None)

    st.caption(
        "Big Data Analytics Lab | Logistic regression + TF-IDF | Trained on 1.5M records"
    )

# ----------------------------------------------------------------
# Shared header
# ----------------------------------------------------------------
st.title("SentimentScope", icon=":material/analytics:")
st.markdown(
    "End-to-end :violet[**PySpark MLlib**] pipeline for "
    ":blue[**binary sentiment classification**]: text cleaning, tokenization, "
    "stop-word removal, :violet[**TF-IDF vectorization**], and "
    ":blue[**logistic regression**] trained on :green[**1.5 million**] "
    "labeled sentences."
)
st.markdown(
    ":blue-badge[PySpark MLlib] :violet-badge[TF-IDF] "
    ":green-badge[Logistic regression] :orange-badge[Binary classification]"
    + (
        " :green-badge[:material/check_circle: Engine ready]"
        if engine_ready
        else " :red-badge[:material/error: Engine offline]"
    )
)

if not engine_ready:
    st.error(
        "Ensure that `sentiment_tfidf_logistic_pipeline.zip` or "
        "`models/sentiment_tfidf_logistic_pipeline/` is present and that Java 8, 11, "
        "or 17 is installed and available on this machine.",
        title="The PySpark model pipeline could not be loaded",
        icon=":material/error:",
    )

# ----------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------
pages = st.navigation(
    [
        st.Page(
            "app_pages/analyzer.py",
            title="Single analysis",
            icon=":material/search:",
            default=True,
        ),
        st.Page(
            "app_pages/batch.py",
            title="Batch analysis",
            icon=":material/dataset:",
        ),
        st.Page(
            "app_pages/evaluation.py",
            title="Model evaluation",
            icon=":material/query_stats:",
        ),
        st.Page(
            "app_pages/pipeline.py",
            title="Pipeline specs",
            icon=":material/schema:",
        ),
    ]
)
pages.run()

st.space("large")
st.caption(
    "SentimentScope | Big Data Analytics final project | Built with PySpark MLlib "
    "and Streamlit",
    text_alignment="center",
)
