"""Meta-learning for automated anomaly detection model selection.

For each of the 58 NAB time series this script:
  1. labels points inside NAB anomaly windows,
  2. extracts 9 statistical meta-features,
  3. scores the series with 4 unsupervised detectors (Isolation Forest, PCA
     reconstruction error, One-Class SVM, K-means) and reads the autoencoder
     F1 scores computed in Autoencoder.ipynb from ae_performance.csv,
  4. trains a Random Forest meta-learner that predicts the best detector and
     evaluates it with leave-one-dataset-out cross-validation.

Run scripts/clean_nab.py first (it writes data/nab_cleaned/), then:
  python meta_learner.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf, adfuller

#### Paths (resolved relative to this file, so the script runs from any folder) ####

REPO_ROOT = Path(__file__).resolve().parent
NAB_ROOT = REPO_ROOT / "data" / "NAB"            # git submodule: numenta/NAB
DATA_DIR = REPO_ROOT / "data" / "nab_cleaned"    # output of scripts/clean_nab.py
LABELS_PATH = NAB_ROOT / "labels" / "combined_windows.json"
AE_PERF_PATH = REPO_ROOT / "ae_performance.csv"  # autoencoder F1 scores from Autoencoder.ipynb


def apply_labels(df: pd.DataFrame, key: str, labels: dict) -> pd.DataFrame:
    df = df.copy()

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # if the index is the timestamp:
        df.index = pd.to_datetime(df.index)
        df["timestamp"] = df.index

    # Start with all zeros
    df["label"] = 0

    # Labels dict uses keys WITH '.csv'
    label_key = key if key.endswith(".csv") else key + ".csv"

    windows = labels.get(label_key, [])

    for w in windows:
        # handle either list ['start', 'end'] or dict {'start':..., 'end':...}
        if isinstance(w, dict):
            start = pd.to_datetime(w.get("start") or w.get("begin") or w.get("start_time"))
            end = pd.to_datetime(w.get("end") or w.get("stop") or w.get("end_time"))
        else:
            # assume list/tuple [start, end]
            start = pd.to_datetime(w[0])
            end = pd.to_datetime(w[1])

        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df.loc[mask, "label"] = 1

    return df


# Meta-feature extraction
def compute_meta_features(df):
    values = df["value"].values

    # basic stats
    mean_val = np.mean(values)
    std_val = np.std(values)
    skew_val = skew(values)
    kurt_val = kurtosis(values)

    # autocorrelation
    acf_vals = acf(values, nlags=5, fft=True)
    acf1 = acf_vals[1]  # lag-1 autocorrelation

    # stationarity (ADF test). adfuller raises on a constant series
    # (artificialNoAnomaly/art_flatline), so record NaN there; NaNs are
    # filled with 0 before the meta-learner is trained.
    if np.ptp(values) == 0:
        adf_stat, p_val = np.nan, np.nan
    else:
        adf_stat, p_val, _, _, _, _ = adfuller(values, autolag="AIC")

    # anomaly ratio (fraction of points inside NAB anomaly windows; uses the labels)
    anomaly_ratio = df["label"].mean()

    return {
        "mean": mean_val,
        "std": std_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "acf1": acf1,
        "adf_stat": adf_stat,
        "adf_p": p_val,
        "anomaly_ratio": anomaly_ratio,
        "length": len(values)
    }


###### Model Construction ######
## 5 Unsupervised models
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

#### Isolation Forest base model ####

def fit_isolation_forest_for_dataset(
    df: pd.DataFrame,
    value_col: str = "value",
    n_estimators: int = 200,
    max_samples="auto",
    contamination="auto",
    random_state: int = 42,
):

    # Extract and scale the univariate series
    X = df[[value_col]].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    iso = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    raw_scores = iso.decision_function(X_scaled)
    anomaly_scores = -raw_scores

    scores = pd.Series(
        anomaly_scores,
        index=df.index,
        name="iso_forest_score",
    )
    return scores, iso, scaler


def run_isolation_forest_on_all_datasets(
    processed: dict,
    value_col: str = "value",
    **iso_kwargs,
):

    iso_scores = {}
    iso_models = {}
    iso_scalers = {}

    for key, df in processed.items():
        scores, model, scaler = fit_isolation_forest_for_dataset(
            df,
            value_col=value_col,
            **iso_kwargs,
        )
        iso_scores[key] = scores
        iso_models[key] = model
        iso_scalers[key] = scaler

    return iso_scores, iso_models, iso_scalers


from sklearn.metrics import precision_recall_fscore_support

def scores_to_binary_predictions(scores: pd.Series, top_frac: float = 0.01) -> pd.Series:
    # Quantile threshold
    threshold = scores.quantile(1 - top_frac)
    y_pred = (scores >= threshold).astype(int)
    y_pred.name = "y_pred"
    return y_pred

def evaluate_isolation_forest(
    iso_scores: dict,
    processed: dict,
    top_frac: float = 0.01,
):
    # Generic evaluator (used for every base model despite the name):
    # top `top_frac` of scores -> predicted anomalies, then precision/recall/F1.

    rows = []

    for key, scores in iso_scores.items():
        if key not in processed:
            continue

        df = processed[key]
        if "label" not in df.columns:
            continue

        # y_true from the label column
        y_true = df["label"].astype(int)

        # align to scores index just in case
        y_true = y_true.reindex(scores.index).fillna(0).astype(int)

        # y_pred from scores via top_frac
        y_pred = scores_to_binary_predictions(scores, top_frac=top_frac)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        rows.append({
            "dataset": key,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(y_true.sum()),
        })

    if not rows:
        # avoid KeyError; return empty but well-formed frame
        return pd.DataFrame(columns=["precision", "recall", "f1", "support"])

    perf_df = pd.DataFrame(rows).set_index("dataset")
    return perf_df


from sklearn.decomposition import PCA
#### PCA Reconstruction Error base model ####

def fit_pca_reconstruction_for_dataset(
    df: pd.DataFrame,
    value_col: str = "value",
    window_size: int = 20,
    n_components: int = 1,
    random_state: int = 42,
):

    values = df[value_col].to_numpy(dtype=float)
    n = len(values)
    if n < window_size:
        # Not enough points for a full window, give flat scores
        scores = pd.Series(
            np.zeros(n),
            index=df.index,
            name="pca_recon_error",
        )
        return scores, None, None

    # Standardize the series first
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1)).ravel()

    # Build sliding-window matrix X
    windows = []
    for i in range(n - window_size + 1):
        windows.append(values_scaled[i : i + window_size])
    X = np.array(windows)

    # Fit PCA
    n_components_eff = min(n_components, window_size)
    pca = PCA(n_components=n_components_eff, random_state=random_state)
    X_proj = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_proj)

    # Reconstruction error per window (MSE over the window)
    recon_error = ((X - X_recon) ** 2).mean(axis=1)

    # Map window-level error back to time index
    full_scores = np.zeros(n)
    full_scores[:] = np.nan

    offset = window_size - 1
    for i, err in enumerate(recon_error):
        t_idx = offset + i
        full_scores[t_idx] = err

    # Fill first (window_size-1) points with the first valid error
    first_valid = np.nanmin(full_scores)
    full_scores[:offset] = first_valid

    scores = pd.Series(
        full_scores,
        index=df.index,
        name="pca_recon_error",
    )

    return scores, pca, scaler

def run_pca_reconstruction_on_all_datasets(
    processed: dict,
    value_col: str = "value",
    window_size: int = 20,
    n_components: int = 1,
):

    pca_scores = {}
    pca_models = {}
    pca_scalers = {}

    for key, df in processed.items():
        scores, pca, scaler = fit_pca_reconstruction_for_dataset(
            df,
            value_col=value_col,
            window_size=window_size,
            n_components=n_components,
        )
        pca_scores[key] = scores
        pca_models[key] = pca
        pca_scalers[key] = scaler

    return pca_scores, pca_models, pca_scalers


from sklearn.svm import OneClassSVM
#### One-Class SVM base model ####

def fit_ocsvm_for_dataset(
    df: pd.DataFrame,
    value_col: str = "value",
    kernel: str = "rbf",
    nu: float = 0.01,
    gamma: str = "scale",
):

    values = df[[value_col]].to_numpy(dtype=float)

    # Scale to zero mean / unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(values)

    ocsvm = OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma=gamma,
    )
    ocsvm.fit(X_scaled)

    # decision_function: positive = inlier, negative = outlier
    raw_scores = ocsvm.decision_function(X_scaled).ravel()

    # Flip so higher = more anomalous (consistent with IF/PCA)
    anomaly_scores = -raw_scores

    scores = pd.Series(
        anomaly_scores,
        index=df.index,
        name="ocsvm_score",
    )

    return scores, ocsvm, scaler

def run_ocsvm_on_all_datasets(
    processed: dict,
    value_col: str = "value",
    kernel: str = "rbf",
    nu: float = 0.01,
    gamma: str = "scale",
):

    ocsvm_scores = {}
    ocsvm_models = {}
    ocsvm_scalers = {}

    for key, df in processed.items():
        scores, model, scaler = fit_ocsvm_for_dataset(
            df,
            value_col=value_col,
            kernel=kernel,
            nu=nu,
            gamma=gamma,
        )
        ocsvm_scores[key] = scores
        ocsvm_models[key] = model
        ocsvm_scalers[key] = scaler

    return ocsvm_scores, ocsvm_models, ocsvm_scalers


from sklearn.cluster import KMeans
#### K-Means clustering base model ####

def fit_kmeans_for_dataset(
    df: pd.DataFrame,
    value_col: str = "value",
    n_clusters: int = 3,
    random_state: int = 42,
):

    # 1D values -> shape (n_samples, 1)
    values = df[[value_col]].to_numpy(dtype=float)

    # Standardize before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(values)

    # Fit KMeans
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    km.fit(X_scaled)

    # Distance from each point to its cluster center
    labels = km.labels_
    centers = km.cluster_centers_
    diffs = X_scaled - centers[labels]
    dist_to_center = np.linalg.norm(diffs, axis=1)  # shape (n_samples,)

    scores = pd.Series(
        dist_to_center,
        index=df.index,
        name="kmeans_score",
    )

    return scores, km, scaler

def run_kmeans_on_all_datasets(
    processed: dict,
    value_col: str = "value",
    n_clusters: int = 3,
    random_state: int = 42,
):

    kmeans_scores = {}
    kmeans_models = {}
    kmeans_scalers = {}

    for key, df in processed.items():
        scores, model, scaler = fit_kmeans_for_dataset(
            df,
            value_col=value_col,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        kmeans_scores[key] = scores
        kmeans_models[key] = model
        kmeans_scalers[key] = scaler

    return kmeans_scores, kmeans_models, kmeans_scalers


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


def main():
    #### Loading/sorting Datasets ####

    # Load all cleaned time-series CSV datasets
    csv_files = list(DATA_DIR.glob("**/*.csv"))
    if not csv_files:
        raise SystemExit(
            f"No cleaned CSVs found in {DATA_DIR}. Run first:\n"
            "  python scripts/clean_nab.py --input-dir data/NAB/data --output-dir data/nab_cleaned"
        )

    datasets = {}
    for f in csv_files:
        rel = f.relative_to(DATA_DIR)
        key = rel.with_suffix("")
        key = str(key).replace("\\", "/")  # ensure forward slashes
        df = pd.read_csv(f)
        datasets[key] = df

    print("Loaded datasets:", len(datasets))  # should be 58


    # Load anomaly windows
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    print("Labelled files:", len(labels))  # should be 58

    #####
    print("Label keys:", list(labels.keys())[:20])
    for k in list(labels.keys())[:5]:
        print("\nKEY:", k)
        print(labels[k])
    #####


    # Build processed dataset dictionary
    processed = {}
    for key, df in datasets.items():
        processed[key] = apply_labels(df, key, labels)

    print("Processed datasets:", len(processed))  # should be 58

    # Build meta-feature matrix
    meta_features = {}

    for key, df in processed.items():
        meta_features[key] = compute_meta_features(df)

    meta_df = pd.DataFrame(meta_features).T
    print(meta_df.head())


    #### Isolation Forest base model ####

    iso_scores, iso_models, iso_scalers = run_isolation_forest_on_all_datasets(
        processed,
        value_col="value",
        contamination=0.01,
        n_estimators=200,
    )

    print(list(iso_scores.keys())[:5])
    print(iso_scores[list(iso_scores.keys())[0]].head())


    iso_perf_iforest = evaluate_isolation_forest(
        iso_scores=iso_scores,
        processed=processed,
        top_frac=0.01,
    )

    print(iso_perf_iforest.head())
    print(iso_perf_iforest.describe())


    print("Checking labels per dataset...")
    for key, df in processed.items():
        if "label" not in df.columns:
            print(key, "-> NO 'label' column")
            continue
        ones = int((df["label"] == 1).sum())
        zeros = int((df["label"] == 0).sum())
        print(key, "label==1:", ones, "label==0:", zeros)


    #### PCA Reconstruction Error base model ####

    pca_scores, pca_models, pca_scalers = run_pca_reconstruction_on_all_datasets(
        processed,
        value_col="value",
        window_size=20,
        n_components=1,
    )

    # Quick sanity check
    print(list(pca_scores.keys())[:5])
    print(pca_scores[list(pca_scores.keys())[0]].head())

    pca_perf = evaluate_isolation_forest(
        iso_scores=pca_scores,
        processed=processed,
        top_frac=0.01,
    )

    print(pca_perf.head())
    print(pca_perf.describe())


    #### One-Class SVM base model ####

    ocsvm_scores, ocsvm_models, ocsvm_scalers = run_ocsvm_on_all_datasets(
        processed,
        value_col="value",
        kernel="rbf",
        nu=0.01,
        gamma="scale",
    )

    # Quick sanity check on scores
    print(list(ocsvm_scores.keys())[:5])
    print(ocsvm_scores[list(ocsvm_scores.keys())[0]].head())

    # Evaluate OCSVM using the same evaluation function
    ocsvm_perf = evaluate_isolation_forest(
        iso_scores=ocsvm_scores,
        processed=processed,
        top_frac=0.01,   # same top-1% anomaly threshold for comparability
    )

    print(ocsvm_perf.head())
    print(ocsvm_perf.describe())


    #### K-Means clustering base model ####

    kmeans_scores, kmeans_models, kmeans_scalers = run_kmeans_on_all_datasets(
        processed,
        value_col="value",
        n_clusters=3,
        random_state=42,
    )

    # Quick sanity check
    print(list(kmeans_scores.keys())[:5])
    print(kmeans_scores[list(kmeans_scores.keys())[0]].head())

    # Evaluate K-Means using the same evaluation function (top-1% as anomalies)
    kmeans_perf = evaluate_isolation_forest(
        iso_scores=kmeans_scores,
        processed=processed,
        top_frac=0.01,
    )

    print(kmeans_perf.head())
    print(kmeans_perf.describe())


    #### Insert Google Colab Autoencoder (deep learning) model and performance ####

    # Load Autoencoder F1 scores that were computed on Colab (Autoencoder.ipynb)
    ae_perf = pd.read_csv(AE_PERF_PATH, index_col="dataset")

    print("AE performance (first 5 rows):")
    print(ae_perf.head())


    # Build performance matrix (F1 scores of each base model)

    perf_df = pd.DataFrame({
        "iso_f1":    iso_perf_iforest["f1"],
        "pca_f1":    pca_perf["f1"],
        "ocsvm_f1":  ocsvm_perf["f1"],
        "kmeans_f1": kmeans_perf["f1"],
        "ae_f1":     ae_perf["f1"],
    })

    print("Performance matrix (first 5 rows):")
    print(perf_df.head())

    # Align on dataset name
    combined = meta_df.join(perf_df, how="inner")

    print("Combined meta-features + performance (first 5 rows):")
    print(combined.head())

    # Columns that hold model performances
    model_f1_cols = ["iso_f1", "pca_f1", "ocsvm_f1", "kmeans_f1", "ae_f1"]

    # Best model name per dataset (string label)
    combined["best_model"] = combined[model_f1_cols].idxmax(axis=1)

    # Map to integers for ML models
    model_to_int = {name: i for i, name in enumerate(model_f1_cols)}
    combined["best_model_idx"] = combined["best_model"].map(model_to_int)

    print(combined[["best_model", "best_model_idx"]].head())


    # Feature columns = all the meta-features

    meta_feature_cols = [
        "mean", "std", "skew", "kurtosis",
        "acf1", "adf_stat", "adf_p",
        "anomaly_ratio", "length",
    ]

    # Make a copy so original is not corrupted
    combined_clean = combined.copy()

    # Replace inf / -inf with NaN
    combined_clean[meta_feature_cols] = combined_clean[meta_feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    # Fill any remaining NaNs with 0
    combined_clean[meta_feature_cols] = combined_clean[meta_feature_cols].fillna(0.0)

    # Check that everything is finite
    print("Any NaNs left?", combined_clean[meta_feature_cols].isna().any().any())

    # Build X and y from the cleaned features
    X = combined_clean[meta_feature_cols].values
    y = combined_clean["best_model_idx"].values


    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    # Leave-one-dataset-out CV
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y, cv=loo)

    print(f"Meta-learner LOOCV accuracy: {scores.mean():.3f}")
    print("Per-fold accuracies:", scores)


    # Recreate LOOCV predictions
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fresh RF each fold
        clf_fold = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
        clf_fold.fit(X_train, y_train)

        y_pred = clf_fold.predict(X_test)

        y_true_all.append(int(y_test[0]))
        y_pred_all.append(int(y_pred[0]))

    # Convert to numpy arrays just in case
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Overall LOOCV accuracy (should match scores.mean())
    cm_accuracy = (y_true_all == y_pred_all).mean()
    print(f"LOOCV accuracy from confusion matrix: {cm_accuracy:.3f}")

    # Confusion matrix using fixed label order [0,1,2,3,4]
    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    print("Raw confusion matrix (rows = true, cols = predicted):")
    print(cm)

    # Print with model names instead of numbers
    int_to_model = {v: k for k, v in model_to_int.items()}
    model_labels = [int_to_model[i] for i in labels]

    cm_df = pd.DataFrame(cm, index=model_labels, columns=model_labels)
    print("\nConfusion matrix:")
    print(cm_df)

    print("\nClassification report:")
    print(classification_report(y_true_all, y_pred_all, target_names=model_labels))


    clf.fit(X, y)  # given a new dataset's meta-features, clf predicts which base model (iso/pca/ocsvm/kmeans/ae) will work best.
    return clf


if __name__ == "__main__":
    main()
