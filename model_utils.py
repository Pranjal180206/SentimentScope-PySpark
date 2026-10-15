"""
model_utils.py - Spark MLlib Model Loading and Prediction Module for SentimentScope

Features:
1. Locates extracted pipeline or extracts sentiment_tfidf_logistic_pipeline.zip.
2. Initializes and caches a singleton SparkSession and PipelineModel.
3. Implements exact training-time text cleaning and filler-word removal.
4. Performs single-sentence and batch-sentence sentiment predictions.
5. Employs comprehensive error handling, health checks, and Spark session management.
"""

import os
import sys
import re
import glob
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Global Singletons
_SPARK_SESSION = None
_PIPELINE_MODEL = None
_MODEL_DIR_PATH: Optional[Path] = None

# Filler words explicitly filtered during model training
FILLER_WORDS = {
    "uh", "um", "erm", "hmm", "ah", "actually", "basically", "literally"
}


def _java_executable(java_home: Path) -> Path:
    executable = "java.exe" if os.name == "nt" else "java"
    return java_home / "bin" / executable


def _java_major_version(java_home: Path) -> int:
    """Reads the major Java version from a JDK/JRE release file."""
    try:
        release_lines = (java_home / "release").read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
    except OSError:
        return 0

    for line in release_lines:
        if line.startswith("JAVA_VERSION="):
            version = line.split("=", 1)[1].strip().strip('"')
            parts = version.split(".")
            if parts[0] == "1" and len(parts) > 1:
                parts = parts[1:]
            return int(parts[0]) if parts[0].isdigit() else 0
    return 0


def _find_java_homes() -> List[Path]:
    """Discovers installed Java runtimes in common locations."""
    if os.name == "nt":
        patterns = [
            r"C:\Program Files\Java\jdk*",
            r"C:\Program Files\Java\jre*",
            r"C:\Program Files\Eclipse Adoptium\jdk*",
            r"C:\Program Files\Microsoft\jdk*",
            r"C:\Program Files\Amazon Corretto\jdk*",
            r"C:\Program Files\Zulu\zulu*",
            r"C:\Java\jdk*",
        ]
    else:
        patterns = [
            "/usr/lib/jvm/*",
            "/usr/java/*",
            "/opt/java/*",
        ]

    homes = [Path(path) for pattern in patterns for path in glob.glob(pattern)]
    current_java = os.environ.get("JAVA_HOME", "")
    if current_java:
        homes.insert(0, Path(current_java))
    return [home for home in homes if _java_executable(home).is_file()]


def configure_java_environment() -> Optional[str]:
    """
    Ensures JAVA_HOME points to a usable Java runtime.
    Prefers JAVA_HOME or another discovered Java 17+ installation, then falls
    back to any available runtime for PySpark 3.5 on older Java versions.
    """
    homes = _find_java_homes()
    modern_homes = [home for home in homes if _java_major_version(home) >= 17]

    if modern_homes:
        current_java = os.environ.get("JAVA_HOME", "")
        selected = next(
            (home for home in modern_homes if str(home) == current_java),
            modern_homes[0],
        )
    elif homes:
        selected = homes[0]
    else:
        return os.environ.get("JAVA_HOME") or None

    selected_str = str(selected)
    os.environ["JAVA_HOME"] = selected_str
    bin_dir = str(selected / "bin")
    path_env = os.environ.get("PATH", "")
    if bin_dir not in path_env:
        os.environ["PATH"] = bin_dir + os.pathsep + path_env
    return selected_str


def clean_sentence(sentence: str) -> str:
    """
    Applies the exact text-cleaning and filler-word-removal logic used during model training:
    1. Converts text to lowercase.
    2. Strips URLs (http://, https://, www.).
    3. Strips user mentions (@username).
    4. Strips non-alphabetic characters (preserving single quotes and whitespace).
    5. Tokenizes and removes defined filler words ('uh', 'um', 'actually', etc.).
    Preserves critical sentiment negations like 'no', 'not', 'never'.
    """
    if not isinstance(sentence, str):
        return ""
    text = sentence.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z'\s]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in FILLER_WORDS]
    return " ".join(tokens)


def locate_and_extract_model(base_dir: Optional[Path] = None) -> Path:
    """
    Resolves the saved PipelineModel directory:
    - If extracted folder exists at models/sentiment_tfidf_logistic_pipeline, validates and returns it.
    - If not extracted, locates sentiment_tfidf_logistic_pipeline.zip, extracts it to models/, and validates.
    - Searches standard locations (project root, parent dir, user downloads).
    """
    global _MODEL_DIR_PATH
    if _MODEL_DIR_PATH and _MODEL_DIR_PATH.exists() and _is_valid_model_dir(_MODEL_DIR_PATH):
        return _MODEL_DIR_PATH

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    target_extracted_dir = base_dir / "models" / "sentiment_tfidf_logistic_pipeline"

    # 1. Check if target extracted directory already exists and is valid
    if target_extracted_dir.exists() and _is_valid_model_dir(target_extracted_dir):
        _MODEL_DIR_PATH = target_extracted_dir
        return target_extracted_dir

    # 2. Check alternative local extracted directories
    alt_dirs = [
        base_dir / "sentiment_tfidf_logistic_pipeline",
        base_dir.parent / "models" / "sentiment_tfidf_logistic_pipeline",
        Path(os.path.expanduser("~")) / "Downloads" / "sentiment_tfidf_logistic_pipeline" / "sentiment_tfidf_logistic_pipeline",
    ]
    for alt in alt_dirs:
        if alt.exists() and _is_valid_model_dir(alt):
            # Copy to target directory if needed, or point directly
            _MODEL_DIR_PATH = alt
            return alt

    # 3. Locate ZIP archive to extract
    zip_candidates = [
        base_dir / "sentiment_tfidf_logistic_pipeline.zip",
        base_dir / "models" / "sentiment_tfidf_logistic_pipeline.zip",
        base_dir.parent / "sentiment_tfidf_logistic_pipeline.zip",
        Path(os.path.expanduser("~")) / "Downloads" / "sentiment_tfidf_logistic_pipeline.zip",
    ]

    found_zip: Optional[Path] = None
    for z in zip_candidates:
        if z.exists() and z.is_file():
            found_zip = z
            break

    if not found_zip:
        raise FileNotFoundError(
            "Saved model archive not found. Please ensure 'sentiment_tfidf_logistic_pipeline.zip' "
            "or the extracted 'models/sentiment_tfidf_logistic_pipeline/' directory is present.\n"
            f"Searched in: {[str(p) for p in zip_candidates]}"
        )

    # Extract the archive into target_extracted_dir
    target_extracted_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_extract_dir = base_dir / "models" / "_temp_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(found_zip, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # Check if zip contained root folder sentiment_tfidf_logistic_pipeline/ or directly metadata/stages
        candidate_nested = temp_extract_dir / "sentiment_tfidf_logistic_pipeline"
        if candidate_nested.exists() and _is_valid_model_dir(candidate_nested):
            source_dir = candidate_nested
        elif _is_valid_model_dir(temp_extract_dir):
            source_dir = temp_extract_dir
        else:
            # Search subdirectories for metadata and stages
            matched = [p.parent for p in temp_extract_dir.rglob("metadata") if (p.parent / "stages").exists()]
            if matched:
                source_dir = matched[0]
            else:
                raise ValueError("ZIP archive does not contain a valid Spark MLlib PipelineModel structure.")

        if target_extracted_dir.exists():
            shutil.rmtree(target_extracted_dir, ignore_errors=True)
        shutil.move(str(source_dir), str(target_extracted_dir))

    finally:
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

    if not _is_valid_model_dir(target_extracted_dir):
        raise ValueError(f"Extracted directory {target_extracted_dir} does not contain valid model stages and metadata.")

    _MODEL_DIR_PATH = target_extracted_dir
    return target_extracted_dir


def _is_valid_model_dir(path: Path) -> bool:
    """Verifies that the directory contains required Spark MLlib PipelineModel structure."""
    if not path.is_dir():
        return False
    metadata_dir = path / "metadata"
    stages_dir = path / "stages"
    return metadata_dir.exists() and stages_dir.exists()


def get_spark_session():
    """
    Initializes and caches a single SparkSession.
    Configured for local execution with reduced overhead for Streamlit interactivity.
    """
    global _SPARK_SESSION
    if _SPARK_SESSION is not None:
        try:
            # Check if active
            if not _SPARK_SESSION.sparkContext._jsc.sc().isStopped():
                return _SPARK_SESSION
        except Exception:
            _SPARK_SESSION = None

    # Ensure Java environment is ready
    configure_java_environment()

    # Set PySpark Python interpreter
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    try:
        from pyspark.sql import SparkSession

        # Build clean SparkSession
        builder = (
            SparkSession.builder
            .appName("SentimentScope")
            .master("local[1]")  # Lightweight local execution
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
        )
        
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        _SPARK_SESSION = spark
        return _SPARK_SESSION
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize PySpark Session. Ensure Java (version 8, 11, or 17) is installed and accessible.\n"
            f"Error details: {str(e)}"
        )


def load_pipeline_model():
    """
    Loads the trained PySpark PipelineModel using PipelineModel.load().
    Caches the loaded model so subsequent calls reuse the same instance without refitting.
    """
    global _PIPELINE_MODEL
    if _PIPELINE_MODEL is not None:
        return _PIPELINE_MODEL

    model_dir = locate_and_extract_model()
    
    # Ensure Spark session is running
    spark = get_spark_session()

    try:
        from pyspark.ml import PipelineModel
        # Spark paths on Windows require normalized or forward slash paths
        clean_path = str(model_dir).replace("\\", "/")
        _PIPELINE_MODEL = PipelineModel.load(clean_path)
        return _PIPELINE_MODEL
    except Exception as e:
        raise RuntimeError(
            f"Failed to load fitted PipelineModel from {model_dir}.\n"
            f"Error details: {str(e)}"
        )


def get_model_health_status() -> Dict[str, Any]:
    """
    Performs a lightweight diagnostic check on Java, Spark, and model availability.
    Returns status metadata without raising uncaught exceptions.
    """
    status = {
        "status": "Unknown",
        "java_home": os.environ.get("JAVA_HOME", "Not Configured"),
        "spark_initialized": False,
        "spark_version": "N/A",
        "model_loaded": False,
        "model_path": "N/A",
        "error_message": None,
    }

    try:
        java_home = configure_java_environment()
        status["java_home"] = java_home or "Not Configured"

        model_path = locate_and_extract_model()
        status["model_path"] = str(model_path)

        spark = get_spark_session()
        status["spark_initialized"] = True
        status["spark_version"] = spark.version

        model = load_pipeline_model()
        status["model_loaded"] = True
        status["status"] = "Ready"
        status["pipeline_stages"] = [stage.__class__.__name__ for stage in model.stages]
    except Exception as e:
        status["status"] = "Error"
        status["error_message"] = str(e)

    return status


def predict_single_sentiment(sentence: str) -> Dict[str, Any]:
    """
    Runs sentiment classification on a single input string:
    1. Applies training-time clean_sentence().
    2. Builds a 1-row Spark DataFrame with 'sentence' column.
    3. Transforms using saved PipelineModel.
    4. Extracts prediction, class 0 (Negative) prob, and class 1 (Positive) prob.
    """
    raw_sentence = sentence.strip() if isinstance(sentence, str) else ""
    if not raw_sentence:
        return {
            "error": "Input text cannot be empty.",
            "is_empty": True
        }

    cleaned = clean_sentence(raw_sentence)
    if not cleaned.strip():
        return {
            "original_text": raw_sentence,
            "cleaned_text": "(no informative tokens after cleaning)",
            "prediction": 0,
            "sentiment": "Neutral / Undetermined",
            "positive_probability": 0.5000,
            "negative_probability": 0.5000,
            "explanation": (
                "After filtering filler words and non-alphabetic characters, no recognizable tokens remained. "
                "The model assigns balanced 50/50 prior likelihood."
            ),
            "is_empty": False,
            "zero_tokens": True
        }

    spark = get_spark_session()
    model = load_pipeline_model()

    # Create Spark DataFrame with 'sentence' column
    df = spark.createDataFrame([(cleaned,)], ["sentence"])

    # Predict using saved pipeline
    predictions = model.transform(df)
    first_row = predictions.select("prediction", "probability").first()

    pred_class = int(first_row["prediction"])
    probabilities = first_row["probability"].toArray().tolist()
    neg_prob = float(probabilities[0])
    pos_prob = float(probabilities[1])

    sentiment_label = "Positive" if pred_class == 1 else "Negative"

    # Generate a neutral, evidence-based interpretation
    explanation = _generate_explanation(cleaned, sentiment_label, pos_prob, neg_prob)

    return {
        "original_text": raw_sentence,
        "cleaned_text": cleaned,
        "prediction": pred_class,
        "sentiment": sentiment_label,
        "positive_probability": round(pos_prob, 4),
        "negative_probability": round(neg_prob, 4),
        "positive_percentage": round(pos_prob * 100, 2),
        "negative_percentage": round(neg_prob * 100, 2),
        "explanation": explanation,
        "is_empty": False,
        "zero_tokens": False
    }


def predict_batch_sentiment(sentences: List[str]) -> pd.DataFrame:
    """
    Performs vectorised sentiment classification across a list of sentences in a single Spark DataFrame.
    Returns a clean Pandas DataFrame suitable for display, charting, and CSV export.
    """
    # Filter empty lines while tracking original line indices
    indexed_data = []
    for idx, s in enumerate(sentences):
        stripped = s.strip() if isinstance(s, str) else ""
        if stripped:
            cleaned = clean_sentence(stripped)
            # Spark requires non-empty string for CountVectorizer
            spark_input = cleaned if cleaned.strip() else "neutral"
            indexed_data.append((idx + 1, stripped, cleaned, spark_input))

    if not indexed_data:
        return pd.DataFrame(columns=[
            "Index", "Original Sentence", "Cleaned Sentence",
            "Predicted Sentiment", "Positive Probability (%)", "Negative Probability (%)",
            "Positive Prob", "Negative Prob", "Prediction"
        ])

    spark = get_spark_session()
    model = load_pipeline_model()

    # Process all rows in a single DataFrame pass
    df_input = spark.createDataFrame(
        [(item[0], item[1], item[2], item[3]) for item in indexed_data],
        ["index", "original_sentence", "cleaned_sentence", "sentence"]
    )

    predictions = model.transform(df_input).select(
        "index", "original_sentence", "cleaned_sentence", "prediction", "probability"
    ).collect()

    rows = []
    for row in predictions:
        pred_class = int(row["prediction"])
        probs = row["probability"].toArray().tolist()
        neg_p = float(probs[0])
        pos_p = float(probs[1])
        sentiment = "Positive" if pred_class == 1 else "Negative"

        rows.append({
            "Index": row["index"],
            "Original Sentence": row["original_sentence"],
            "Cleaned Sentence": row["cleaned_sentence"],
            "Predicted Sentiment": sentiment,
            "Positive Probability (%)": round(pos_p * 100, 2),
            "Negative Probability (%)": round(neg_p * 100, 2),
            "Positive Prob": round(pos_p, 4),
            "Negative Prob": round(neg_p, 4),
            "Prediction": pred_class
        })

    res_df = pd.DataFrame(rows).sort_values("Index").reset_index(drop=True)
    return res_df


def _generate_explanation(cleaned_text: str, sentiment: str, pos_prob: float, neg_prob: float) -> str:
    """
    Provides a short, neutral explanation based on model-estimated probabilities without overclaiming certainty.
    """
    tokens = cleaned_text.split()
    token_count = len(tokens)
    margin = abs(pos_prob - neg_prob)

    if margin > 0.6:
        strength = "strong"
    elif margin > 0.25:
        strength = "moderate"
    else:
        strength = "slight"

    explanation = (
        f"The Logistic Regression model detected a {strength} {sentiment.lower()} orientation "
        f"across the {token_count} processed token(s). "
        f"Model-estimated class probability: {pos_prob * 100:.1f}% Positive vs {neg_prob * 100:.1f}% Negative."
    )
    return explanation


def stop_spark_session():
    """Gracefully terminates the background SparkSession."""
    global _SPARK_SESSION
    if _SPARK_SESSION is not None:
        try:
            _SPARK_SESSION.stop()
        except Exception:
            pass
        _SPARK_SESSION = None
