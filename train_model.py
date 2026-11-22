import csv
import io
import json
import platform
import sys
import zipfile
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ---------- Paths and constants ----------
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

# Official UCI Machine Learning Repository download link
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TOP_N_WORDS = 15


def banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ======================================================================
# STEP 1 CODE (unchanged): download, load, explore
# ======================================================================
def print_manual_instructions():
    banner("MANUAL DOWNLOAD INSTRUCTIONS")
    print(f"1. Open: {DATA_PAGE}")
    print("2. Click 'Download' and unzip the file.")
    print("3. Copy the file named 'SMSSpamCollection' (no extension) into:")
    print(f"   {DATA_FILE.parent}")
    print("4. Run this script again.\n")


def download_dataset():
    """Download and extract the UCI zip. Returns True on success."""
    print(f"Downloading dataset from UCI...\n  {DATA_URL}")
    try:
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            zip_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            DATA_FILE.write_bytes(zf.read("SMSSpamCollection"))

        print(f"Saved to {DATA_FILE}")
        return True
    except Exception as error:  # network error, bad zip, missing file, etc.
        print(f"Automatic download failed: {error}")
        return False


def load_raw_data():
    """Read the tab-separated file (no header: label<TAB>text) into a DataFrame.
    Returns None if the file is unavailable."""
    if not DATA_FILE.exists():
        if not download_dataset():
            print_manual_instructions()
            return None

    # quoting=QUOTE_NONE: SMS texts contain quote characters that would
    # confuse the default CSV parser.
    read_args = dict(
        sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE
    )
    try:
        return pd.read_csv(DATA_FILE, encoding="utf-8", **read_args)
    except UnicodeDecodeError:
        return pd.read_csv(DATA_FILE, encoding="latin-1", **read_args)


def explore(df):
    """Print a quick summary of the raw dataset (only with --explore)."""
    banner("DATASET EXPLORATION (raw data)")
    print(f"Shape (rows, columns): {df.shape}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())
    counts = df["label"].value_counts()
    percents = (df["label"].value_counts(normalize=True) * 100).round(1)
    print("Class distribution:")
    for label in counts.index:
        print(f"  {label:<5} {counts[label]:>5} messages  ({percents[label]}%)")
    print(f"Duplicate messages (extra copies): {df.duplicated(subset='text').sum()}")


# ======================================================================
# STEP 2 CODE: clean, split, train, select, evaluate, save
# ======================================================================
def clean_data(df):
    """Remove bad rows and exact duplicates. Returns (clean_df, info dict)."""
    total_raw = len(df)
    raw_counts = df["label"].value_counts().to_dict()

    df = df.dropna(subset=["label", "text"])

    # Remove exact duplicate messages (keep the first copy). This happens
    # BEFORE the split so identical messages cannot end up in both train
    # and test, which would inflate the scores.
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    duplicates_removed = before - len(df)

    # Tidy whitespace and drop messages that are empty after tidying
    df = df.assign(text=df["text"].str.strip())
    df = df[df["text"] != ""]

    # ham -> 0, spam -> 1  (spam is the "positive" class for all metrics)
    df = df.assign(target=df["label"].map({"ham": 0, "spam": 1}))
    df = df.dropna(subset=["target"]).astype({"target": int})
    df = df.reset_index(drop=True)

    info = {
        "total_messages": int(total_raw),
        "class_counts_raw": {k: int(v) for k, v in raw_counts.items()},
        "duplicates_removed": int(duplicates_removed),
        "messages_after_cleaning": int(len(df)),
        "class_counts_clean": {
            "ham": int((df["target"] == 0).sum()),
            "spam": int((df["target"] == 1).sum()),
        },
        "avg_length_chars": {
            "ham": round(float(df.loc[df["target"] == 0, "text"].str.len().mean()), 1),
            "spam": round(float(df.loc[df["target"] == 1, "text"].str.len().mean()), 1),
        },
    }
    return df, info


def make_pipeline(classifier):
    """TF-IDF + classifier in ONE sklearn Pipeline (identical TF-IDF settings
    for both models, so the comparison is fair)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", classifier),
    ])


def select_settings_with_cv(name, pipeline, param_grid, X_train, y_train):
    """5-fold cross-validation on the TRAINING data only.
    Tries each setting in the small grid, scores it by spam-class F1, then
    refits the best setting on the full training set.
    The TF-IDF step is inside the pipeline, so it is re-fitted inside each
    fold and never sees the validation fold (no data leakage)."""
    print(f"\n[{name}] running {CV_FOLDS}-fold CV on training data...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1",  # F1 of the positive class (label 1 = spam)
        cv=cv,
        refit=True,
    )
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    for _, row in results.iterrows():
        print(f"    {row['params']}  ->  CV spam F1 = {row['mean_test_score']:.4f}")
    print(f"  Best settings: {search.best_params_}  (CV spam F1 = {search.best_score_:.4f})")
    return search


def evaluate_on_test(model, X_test, y_test):
    """Compute the final test-set metrics (spam = positive class)."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        # Layout: [[true ham, false spam], [missed spam, true spam]]
        "confusion_matrix": cm.tolist(),
    }


def top_words(model, n=TOP_N_WORDS):
    """Most informative words, read from the trained model's real weights.
    Logistic Regression: the coefficient of each word.
    Naive Bayes: log P(word | spam) - log P(word | ham).
    Positive weight -> pushes toward spam, negative -> toward ham."""
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    words = np.array(vectorizer.get_feature_names_out())

    if hasattr(clf, "coef_"):  # Logistic Regression
        weights = clf.coef_[0]
    else:  # Multinomial Naive Bayes
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    order = np.argsort(weights)
    spam_idx = order[::-1][:n]   # largest weights
    ham_idx = order[:n]          # smallest weights
    as_list = lambda idx: [
        {"word": str(words[i]), "weight": round(float(weights[i]), 4)} for i in idx
    ]
    return as_list(spam_idx), as_list(ham_idx)


def print_results_table(results, selected):
    print("\n" + "-" * 78)
    print(f"{'Model':<22}{'CV F1':>8}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    print("-" * 78)
    for key, r in results.items():
        mark = "  <-- selected" if key == selected else ""
        print(
            f"{key:<22}{r['cv_f1']:>8.4f}{r['accuracy']:>10.4f}"
            f"{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}{mark}"
        )
    print("-" * 78)
    print("(CV F1 = 5-fold CV on training data, used for selection."
          "  The other columns = test set.)")


def main():
    # ---- 1. Load ----
    banner("STEP 1/6: LOAD DATA")
    df_raw = load_raw_data()
    if df_raw is None:
        return  # instructions already printed; exit cleanly
    print(f"Loaded {len(df_raw)} raw messages.")
    if "--explore" in sys.argv:
        explore(df_raw)

    # ---- 2. Clean ----
    banner("STEP 2/6: CLEAN DATA")
    df, info = clean_data(df_raw)
    print(f"Duplicate messages removed : {info['duplicates_removed']}")
    print(f"Messages after cleaning    : {info['messages_after_cleaning']}")
    print(f"  ham : {info['class_counts_clean']['ham']}")
    print(f"  spam: {info['class_counts_clean']['spam']}")

    # ---- 3. Split ----
    banner("STEP 3/6: TRAIN / TEST SPLIT (80/20, stratified, seed 42)")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"],
        test_size=TEST_SIZE, stratify=df["target"], random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train)} messages  (spam: {int(y_train.sum())})")
    print(f"Test : {len(X_test)} messages  (spam: {int(y_test.sum())})")
    info["train_size"] = int(len(X_train))
    info["test_size"] = int(len(X_test))
    info["train_class_counts"] = {"ham": int((y_train == 0).sum()), "spam": int(y_train.sum())}
    info["test_class_counts"] = {"ham": int((y_test == 0).sum()), "spam": int(y_test.sum())}

    # ---- 4. Cross-validation on TRAINING data only ----
    banner("STEP 4/6: MODEL SELECTION (5-fold CV on training data only)")
    candidates = {
        "naive_bayes": (
            make_pipeline(MultinomialNB()),
            {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
        ),
        "logistic_regression": (
            make_pipeline(LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
            {"clf__C": [1, 10, 100]},
        ),
    }
    searches = {}
    for name, (pipeline, grid) in candidates.items():
        searches[name] = select_settings_with_cv(name, pipeline, grid, X_train, y_train)

    # Selection uses ONLY the cross-validation score. No test data involved.
    selected = max(searches, key=lambda name: searches[name].best_score_)
    print(f"\nSelected model (highest CV spam F1): {selected}")

    # ---- 5. Final evaluation on the test set (used once, for reporting) ----
    banner("STEP 5/6: FINAL TEST-SET EVALUATION")
    results = {}
    for name, search in searches.items():
        metrics = evaluate_on_test(search.best_estimator_, X_test, y_test)
        metrics["cv_f1"] = float(search.best_score_)
        metrics["best_params"] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        results[name] = metrics
    print_results_table(results, selected)

    cm = results[selected]["confusion_matrix"]
    print(f"\nConfusion matrix of selected model ({selected}):")
    print(f"                    predicted ham   predicted spam")
    print(f"  actual ham        {cm[0][0]:>13}   {cm[0][1]:>14}")
    print(f"  actual spam       {cm[1][0]:>13}   {cm[1][1]:>14}")
    print(f"  (false alarms: {cm[0][1]} legit messages flagged;"
          f" missed spam: {cm[1][0]})")

    # ---- 6. Save ----
    banner("STEP 6/6: SAVE MODEL AND METRICS")
    best_model = searches[selected].best_estimator_
    spam_words, ham_words = top_words(best_model)

    metrics_out = {
        "dataset": info,
        "models": results,
        "selected_model": selected,
        "selection_criterion": f"mean spam-class F1 from {CV_FOLDS}-fold CV on training data only",
        "confusion_matrix": results[selected]["confusion_matrix"],
        "top_spam_words": spam_words,
        "top_ham_words": ham_words,
        "trained_with": {
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "random_state": RANDOM_STATE,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        },
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"Saved model   -> {MODEL_FILE}")
    print(f"Saved metrics -> {METRICS_FILE}")

    # ---- Reload check: prove the saved file really works ----
    reloaded = joblib.load(MODEL_FILE)
    same = bool((reloaded.predict(X_test) == best_model.predict(X_test)).all())
    print(f"Reload check: saved model gives identical test predictions -> {same}")

    print("\nTop words pushing toward SPAM:", ", ".join(w["word"] for w in spam_words[:10]))
    print("Top words pushing toward HAM :", ", ".join(w["word"] for w in ham_words[:10]))
    print("\nStep 2 complete. Paste this output back before we build the UI.")


if __name__ == "__main__":
    main()