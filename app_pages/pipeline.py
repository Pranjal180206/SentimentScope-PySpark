"""Model and pipeline specifications."""

import streamlit as st

st.header("Pipeline specifications", icon=":material/schema:", divider="orange")
st.caption(
    "Architecture of the saved PySpark MLlib PipelineModel and the preprocessing "
    "applied before scoring."
)

spec_col, stages_col = st.columns(2)

with spec_col:
    with st.container(border=True, height="stretch"):
        st.subheader("Model specifications")
        st.markdown(
            ":blue-badge[Spark MLlib] :orange-badge[Binary classification]"
        )
        st.markdown(
            "- **Classifier:** Logistic regression (Spark MLlib)\n"
            "- **Model class:** `LogisticRegressionModel`\n"
            "- **Hyperparameters:** `maxIter = 10`, `regParam = 0.01`\n"
            '- **Columns:** `labelCol = "label"`, `featuresCol = "features"`\n'
            "- **Labels:** `0 = Negative`, `1 = Positive`\n"
            "- **Split:** 70 percent train / 30 percent test (seed 42)\n"
            "- **Training records:** 1,066,861\n"
            "- **Testing records:** 456,992"
        )

with stages_col:
    with st.container(border=True, height="stretch"):
        st.subheader("Feature engineering pipeline")
        st.markdown(
            ":violet-badge[TF-IDF] :green-badge[50,000-term vocabulary]"
        )
        st.markdown(
            "1. **Text cleaning (external stage):** lowercasing, URL and mention "
            "removal, filler-word filtering, non-alphabetic character stripping.\n"
            "2. **Tokenizer (stage 0):** splits the cleaned sentence into tokens.\n"
            "3. **StopWordsRemover (stage 1):** removes English stop words while "
            "preserving negations such as `no`, `not`, and `nor`.\n"
            "4. **CountVectorizer (stage 2):** `vocabSize = 50,000`, `minDF = 2.0`.\n"
            "5. **IDF (stage 3):** computes inverse document frequency weights.\n"
            "6. **LogisticRegression (stage 4):** produces class margins and "
            "probabilities."
        )

st.subheader("Integrity and non-retraining guarantees")
st.markdown(
    "- The application loads the fitted `PipelineModel` from "
    "`sentiment_tfidf_logistic_pipeline`; the vocabulary, IDF weights, and "
    "regression coefficients are frozen and reused as-is.\n"
    "- Preprocessing mirrors the training notebook exactly to prevent "
    "feature-distribution skew.\n"
    "- No pipeline stage is refit or fine-tuned at serving time."
)
