# SentimentScope: PySpark-Powered Sentiment Analysis

**SentimentScope** is an interactive sentiment analysis dashboard built with **Streamlit**, **PySpark**, and **Spark MLlib**. It classifies text as **Positive** or **Negative** using a trained TF-IDF + Logistic Regression pipeline fitted on more than **1.5 million** sentences.

The app loads the fitted `PipelineModel` directly, so the vocabulary, IDF weights, and regression coefficients are never refit at serving time.

---

## Features

- **Single sentence analysis:** Instant predictions with positive/negative probabilities, preprocessed token inspection, and a short model-based explanation.
- **Example prompts:** One-click sample sentences covering positive, negative, and ambiguous inputs.
- **Batch analysis:** Classifies many sentences in one Spark DataFrame execution, with summary metrics, a distribution chart, a detailed table, and CSV export.
- **Evaluation dashboard:** Displays held-out test-set metrics, per-class precision/recall/F1, and the confusion matrix.
- **Pipeline specifications:** Documents the exact preprocessing and MLlib stages used during training.
- **Polished theming:** Streamlit light/dark themes defined in `.streamlit/config.toml`.

---

## Model Performance

Evaluated on the held-out 30% test split (456,992 records), trained on 1,066,861 records:

| Metric | Value |
| :--- | ---: |
| Accuracy | **77.77%** |
| ROC-AUC | **0.8489** |
| Positive precision / recall / F1 | 0.7669 / 0.7935 / 0.7800 |
| Negative precision / recall / F1 | 0.7891 / 0.7621 / 0.7754 |

Full metrics and confusion-matrix counts are stored in `sentiment_evaluation_results.json`.

---

## Tech Stack

- **Application:** Streamlit, Altair, pandas
- **Machine learning:** PySpark MLlib 3.5.x
- **Pipeline stages:** Tokenizer, StopWordsRemover, CountVectorizer (50,000-term vocabulary), IDF, Logistic Regression
- **Training notebook:** Google Colab / Jupyter

---

## Project Structure

```text
sentiment-scope/
├── app.py                                                  # Entrypoint: layout and navigation
├── app_pages/
│   ├── analyzer.py                                         # Single sentence analysis
│   ├── batch.py                                            # Batch analysis and CSV export
│   ├── evaluation.py                                       # Evaluation dashboard
│   └── pipeline.py                                         # Pipeline specifications
├── model_utils.py                                          # Spark session, model loading, cleaning, inference
├── sentiment_evaluation_results.json                       # Held-out test metrics
├── Sentiment_Analysis_TFIDF_LogisticRegression_PySpark.ipynb  # Training notebook
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml                                         # Theme configuration
└── models/
    └── sentiment_tfidf_logistic_pipeline/                  # Model artifact (not committed)
```

Model binaries are intentionally excluded from version control. See the next section.

---

## Trained Model

The repository does **not** include the trained pipeline archive or the training dataset. The app looks for the model in this order:

1. `models/sentiment_tfidf_logistic_pipeline/` as an extracted Spark model directory.
2. `sentiment_tfidf_logistic_pipeline/` in the project root.
3. `sentiment_tfidf_logistic_pipeline.zip` in the project root, in `models/`, in the parent directory, or in `~/Downloads/`.

If only the ZIP is available, the app extracts it to `models/sentiment_tfidf_logistic_pipeline/` on first launch. To regenerate the artifact, run the included training notebook on Colab (it expects `train_data.csv` with `sentence,sentiment` columns and labels `0`/`1`).

---

## Quickstart

### Prerequisites

- **Python:** 3.10, 3.11, or 3.12 (64-bit)
- **Java:** 8, 11, or 17. Set `JAVA_HOME` if Java is not on your `PATH`.
- **RAM:** At least 2 GB free for the local Spark session.

### 1. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Provide the model

Place `sentiment_tfidf_logistic_pipeline.zip` in the project root, or pre-extract it to `models/sentiment_tfidf_logistic_pipeline/`.

### 4. Run the app

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Testing Guidelines

Try these sentences in the **Single sentence analysis** view:

| Sentence | Expected Sentiment | Notes |
| :--- | :--- | :--- |
| *"I absolutely love this product. It works perfectly."* | **Positive** | Strong positive tokens (`love`, `perfectly`). |
| *"This is terrible and I regret buying it."* | **Negative** | Strong negative tokens (`terrible`, `regret`). |
| *"The product is okay, nothing special."* | **Negative / Mild** | Ambivalent text with a smaller probability margin. |
| *"I am not happy with the service."* | **Negative** | Negation (`not`) is preserved during preprocessing. |

---

## Training Notebook

`Sentiment_Analysis_TFIDF_LogisticRegression_PySpark.ipynb` contains the end-to-end training workflow:

1. Parse and clean `train_data.csv`.
2. Remove filler words while preserving negations such as `no`, `not`, and `nor`.
3. Split the data 70/30 with seed 42.
4. Fit the TF-IDF and Logistic Regression pipeline.
5. Evaluate on the test split and save the fitted `PipelineModel`.

The notebook uses Colab paths (`/content/...`) and downloads the trained model as a ZIP archive.

---

## Troubleshooting

1. **`JAVA_HOME` or Java version errors:**
   - PySpark 3.5 supports Java 8, 11, and 17.
   - Point `JAVA_HOME` at a JDK/JRE installation:
     ```powershell
     $env:JAVA_HOME = "C:\Path\To\Java"
     ```
   - The app also detects common Windows JDK locations automatically.

2. **Windows hostname or networking errors:**
   - `model_utils.py` binds Spark to `127.0.0.1` and disables the Spark UI, which avoids hostname parsing issues on Windows machines.

3. **Port 8501 already in use:**
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **Model not found:**
   - Confirm that `sentiment_tfidf_logistic_pipeline.zip` or the extracted model directory exists in one of the locations listed above.
   - The extracted directory must contain both `metadata/` and `stages/`.

---

## Deployment Notes

- **Local presentation:** `streamlit run app.py`
- **Docker / cloud:** use a `python:3.11-slim` or `eclipse-temurin:17-jre-jammy` base image, install the Python dependencies, set `JAVA_HOME`, and provide the model artifact through a volume or download step.
- Allow at least 2 GB of RAM for the container because each app process starts a local Spark session.
