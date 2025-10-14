"""
Head Impact Detection using SVM Classifier
==========================================

A comprehensive machine learning pipeline for detecting American football head impacts
using Support Vector Machine classification with advanced feature selection and validation.

Authors: Dishan D (PES1UG23CS196), Samruddhi Patil (PES1UG24CS828)
Course: Machine Learning (UE23CS352A)
Institution: PES University

This implementation includes:
- Statistical feature analysis using Wilcoxon Rank-Sum test
- Automated feature selection with Sequential Forward Selection
- Dual optimization strategies (AUC and F-measure)
- Cross-validation and independent testing
- Comprehensive performance visualization
"""

import matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    make_scorer,
    classification_report,
    recall_score,
    roc_curve,
    precision_recall_curve,
)
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

output_path = "../output"
os.makedirs(output_path, exist_ok=True)

# Dataset Loading and Preprocessing Module


def load_dataset(ds_path, ds_name, silent=False):
    """
    Loads the head impact dataset from Excel file with separate training and testing sheets.
    
    Args:
        ds_path (str): Path to the Excel dataset file
        ds_name (str): Name identifier for the dataset
        silent (bool): If True, suppresses output messages
    
    Returns:
        dict: Dictionary containing X_train, y_train, X_test, y_test DataFrames
    """
    ds = {"path": ds_path, "name": ds_name}

    ds["X_train"] = pd.read_excel(ds_path, sheet_name="Feature Matrix Training")
    ds["y_train"] = pd.read_excel(ds_path, sheet_name="Label Vector Training")

    ds["X_test"] = pd.read_excel(ds_path, sheet_name="Feature Matrix Testing")
    ds["y_test"] = pd.read_excel(ds_path, sheet_name="Label Vector Testing")

    ds["y_train"] = ds["y_train"].squeeze()
    ds["y_test"] = ds["y_test"].squeeze()

    if not silent:
        pad_len = len(ds["name"]) + 2
        print(f"[{ds['name']}] : Loaded dataset successfully")
        print(f"{' ' * pad_len} : - X_train shape = {ds['X_train'].shape}")
        print(f"{' ' * pad_len} : - y_train shape = {ds['y_train'].shape}")
        print(f"{' ' * pad_len} : - X_test shape  = {ds['X_test'].shape}")
        print(f"{' ' * pad_len} : - y_test shape  = {ds['y_test'].shape}")
        print("")

    return ds


# Data Cleaning and Missing Value Handling


def clean_dataset(ds, silent=False):
    """
    Cleans the dataset by removing columns with all NaN values and imputing missing values.
    
    Args:
        ds (dict): Dataset dictionary containing X_train, X_test DataFrames
        silent (bool): If True, suppresses output messages
    
    Returns:
        dict: Updated dataset dictionary with cleaned DataFrames
    """
    ds["X_train"] = ds["X_train"].dropna(axis=1, how="all")
    ds["X_train"] = ds["X_train"].fillna(ds["X_train"].mean())

    ds["X_test"] = ds["X_test"].dropna(axis=1, how="all")
    ds["X_test"] = ds["X_test"].fillna(ds["X_test"].mean())

    if not silent:
        pad_len = len(ds["name"]) + 2
        print(f"[{ds['name']}] : Cleaned dataset successfully")
        print(f"{' ' * pad_len} : - X_train shape = {ds['X_train'].shape}")
        print(f"{' ' * pad_len} : - y_train shape = {ds['y_train'].shape}")
        print(f"{' ' * pad_len} : - X_test shape  = {ds['X_test'].shape}")
        print(f"{' ' * pad_len} : - y_test shape  = {ds['y_test'].shape}")
        print("")

    return ds


# Feature Standardization for SVM Compatibility


def standardize_features(ds, silent=False):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(ds["X_train"])
    X_test_scaled = scaler.transform(ds["X_test"])

    ds["X_train_scaled"] = pd.DataFrame(X_train_scaled, columns=ds["X_train"].columns)
    ds["X_test_scaled"] = pd.DataFrame(X_test_scaled, columns=ds["X_test"].columns)

    if not silent:
        pad_len = len(ds["name"]) + 2
        print(f"[{ds['name']}] : Standardized features successfully")
        print(
            f"{' ' * pad_len} : - X_train_scaled_shape = {ds['X_train_scaled'].shape}]"
        )
        print(
            f"{' ' * pad_len} : - X_test_scaled_shape  = {ds['X_test_scaled'].shape}]"
        )
        print("")

    return ds


# Statistical Feature Significance Analysis


def run_wilcoxon_test(ds, top_n=50, plot=True, silent=False):
    """
    Performs Wilcoxon Rank-Sum test with Bonferroni correction to identify significant features.
    
    Args:
        ds (dict): Dataset dictionary with scaled features and labels
        top_n (int): Number of top features to select based on p-values
        plot (bool): If True, generates visualization of top features
        silent (bool): If True, suppresses output messages
    
    Returns:
        dict: Updated dataset with feature ranking and selected features
    """
    X = ds["X_train_scaled"]
    y = ds["y_train"]

    p_values = []
    for i in range(X.shape[1]):
        group1 = X.loc[y == 1, X.columns[i]]
        group0 = X.loc[y == 0, X.columns[i]]
        _, p = ranksums(group1, group0)
        p_values.append(p)

    _, p_adj, _, _ = multipletests(p_values, method="bonferroni")

    feature_ranking = pd.DataFrame(
        {"feature": X.columns, "p_value": p_values, "p_adj": p_adj}
    ).sort_values("p_adj")

    top_features = feature_ranking["feature"].head(top_n)

    if plot:
        save_path = os.path.join(output_path, "wilcoxon-test.png")
        plt.figure(figsize=(10, 5))
        sns.barplot(
            x="p_adj",
            y="feature",
            data=feature_ranking.head(20),
        )
        plt.title(f"Top 20 Significant Features ({ds['name']})")
        plt.xlabel("Adjusted p-value (Bonferroni)")
        plt.ylabel("Feature Name")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    ds["feature_ranking"] = feature_ranking
    ds["top_features_wilcoxon"] = list(top_features)

    if not silent:
        pad_len = len(ds["name"]) + 2
        print(f"[{ds['name']}] : Ran Wilcoxon Rank-Sum Test successfully")
        print(f"{' ' * pad_len} : Selected top {top_n} features")
        print()

    return ds


# Principal Component Analysis for Dimensionality Assessment


def run_pca_analysis(ds, plot=True, silent=False):
    X = ds["X_train_scaled"]
    top_feats = ds["top_features_wilcoxon"]
    filtered_feats = [
        f
        for f in top_feats
        if ("PSD" in f or "WT" in f) and any(freq in f for freq in ["10", "20", "30"])
    ]

    pca = PCA()
    pca.fit(X[filtered_feats])
    explained = np.cumsum(pca.explained_variance_ratio_)

    if plot:
        save_path = os.path.join(output_path, "pca-analysis.png")
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(explained) + 1), explained * 100, marker="o")
        plt.title(f"Cumulative Variance Explained by PCA ({ds['name']})")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Variance Explained (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    ds["pca_explained_variance"] = explained

    if not silent:
        pad_len = len(ds["name"]) + 2
        print(f"[{ds['name']}] : Ran PCA Analysis successfully")
        print(
            f"{' ' * pad_len} : PCA first 5 components explain {explained[4] * 100:.2f}% variance"
        )
        print()

    return ds


# Feature Correlation Matrix Visualization


def plot_correlation_matrix(ds, silent=False):
    save_path = os.path.join(output_path, "correlation-matrix.png")
    corr_matrix = ds["X_train_scaled"].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.savefig(save_path)
    plt.close()

    if not silent:
        print(f"[{ds['name']}] : Plotted correlation matrix successfully")
        print()

    return ds


# Automated Feature Selection using Sequential Forward Selection


def select_features(ds, scoring_metric="roc_auc", n_features=8, silent=False):
    """
    Runs Sequential Forward Selection (SFS) to choose a compact feature subset.

    To align with Wu et al. (2018), the candidate pool is restricted to low-frequency
    PSD/WT features (10–30 Hz) drawn from the Wilcoxon-ranked top features.

    Args:
        ds (dict): Dataset dictionary with standardized features and labels
        scoring_metric (str): sklearn scoring key (e.g., "roc_auc", "f1")
        n_features (int): Number of features to select (<= 10 per paper guidance)
        silent (bool): Suppress stdout

    Returns:
        dict: Dataset dictionary with key 'selected_features_{scoring_metric}' populated
    """
    X_all = ds["X_train_scaled"]
    y = ds["y_train"]

    # Restrict candidates to PSD/WT and 10/20/30 Hz from Wilcoxon top features
    top_feats = ds.get("top_features_wilcoxon", list(X_all.columns))
    candidate_features = [
        f
        for f in top_feats
        if ("PSD" in f or "WT" in f) and any(freq in f for freq in ["10", "20", "30"])
    ]
    # Fallback: if filtering yields empty set, use Wilcoxon top features
    if len(candidate_features) == 0:
        candidate_features = list(top_feats)

    X = X_all[candidate_features]

    svc = SVC(kernel="rbf", probability=(scoring_metric == "roc_auc"))

    sfs = SequentialFeatureSelector(
        estimator=svc,
        n_features_to_select=n_features,
        direction="forward",
        scoring=scoring_metric,
        cv=10,
        n_jobs=-1,
    )
    sfs.fit(X, y)
    selected_features_mask = sfs.get_support()
    selected_features = list(X.columns[selected_features_mask])

    ds[f"selected_features_{scoring_metric}"] = selected_features

    if not silent:
        pad_len = len(ds["name"]) + 2
        print(f"[{ds['name']}] : Ran SFS with scoring='{scoring_metric}' successfully")
        print(
            f"{' ' * pad_len} : Selected {len(selected_features)} features: {selected_features}"
        )
        print()

    return ds


# Model Training and Cross-Validation Pipeline


def train_and_validate(ds, feature_subset_key, silent=False):
    selected_features = ds[feature_subset_key]
    X = ds["X_train_scaled"][selected_features]
    y = ds["y_train"]

    svc = SVC(kernel="rbf")

    scorers = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "specificity": make_scorer(recall_score, pos_label=0),
    }

    ## Perform 10-fold cross-validation
    scores = cross_validate(svc, X, y, cv=10, scoring=scorers)

    results = {
        "Sensitivity": scores["test_recall"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Accuracy": scores["test_accuracy"].mean(),
        "F-measure": scores["test_f1"].mean(),
        "Specificity": scores["test_specificity"].mean(),
    }

    if not silent:
        print(
            f"--- Collegiate Cross-Validation Results for '{feature_subset_key}' (10-fold CV) ---"
        )
        for metric, value in results.items():
            print(f"  - {metric}: {value:.4f}")
        print()

    ds[f"cv_results_{feature_subset_key}"] = results
    return ds


# Independent Dataset Testing for Generalization Assessment


def test_on_independent_data(ds, feature_subset_key, silent=False):
    selected_features = ds[feature_subset_key]

    X_train = ds["X_train_scaled"][selected_features]
    y_train = ds["y_train"]

    svc = SVC(kernel="rbf").fit(X_train, y_train)

    X_test = ds["X_test_scaled"][selected_features]
    y_test = ds["y_test"]

    y_pred = svc.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    results = {
        "Sensitivity": report.get("1", {}).get("recall", 0),
        "Precision": report.get("1", {}).get("precision", 0),
        "Accuracy": report.get("accuracy", 0),
        "F-measure": report.get("1", {}).get("f1-score", 0),
        "Specificity": report.get("0", {}).get("recall", 0),
    }

    if not silent:
        print(
            f"--- Independent Test Results for '{feature_subset_key}' on Youth Data ---"
        )
        for metric, value in results.items():
            print(f"  - {metric}: {value:.4f}")
        print()

    ds[f"test_results_{feature_subset_key}"] = results
    return ds


# Performance Visualization: ROC and Precision-Recall Curves


def plot_roc_pr_curves(ds, auc_features, f1_features, silent=False):
    save_path = os.path.join(output_path, "roc-precision-recall.png")
    X_train = ds["X_train_scaled"]
    y_train = ds["y_train"]

    svc_auc = SVC(kernel="rbf").fit(X_train[auc_features], y_train)
    svc_f1 = SVC(kernel="rbf").fit(X_train[f1_features], y_train)

    scores_auc = svc_auc.decision_function(X_train[auc_features])
    scores_f1 = svc_f1.decision_function(X_train[f1_features])

    accel_feature = "'lin acc peak 4'"
    scores_accel = X_train[accel_feature]

    ## ROC Curve
    fpr_auc, tpr_auc, _ = roc_curve(y_train, scores_auc)
    fpr_f1, tpr_f1, _ = roc_curve(y_train, scores_f1)
    fpr_accel, tpr_accel, _ = roc_curve(y_train, scores_accel)

    ## Precision-Recall Curve
    precision_auc, recall_auc, _ = precision_recall_curve(y_train, scores_auc)
    precision_f1, recall_f1, _ = precision_recall_curve(y_train, scores_f1)
    precision_accel, recall_accel, _ = precision_recall_curve(y_train, scores_accel)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ## Plot ROC curve
    ax1.plot(fpr_auc, tpr_auc, label="SVM (AUC)")
    ax1.plot(fpr_f1, tpr_f1, label="SVM (F-meas)")
    ax1.plot(fpr_accel, tpr_accel, label="Accel Thresh")
    ax1.set_title("Receiver Operating Characteristics", fontsize=14)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    ax1.grid(alpha=0.3)

    ## Plot precision-recall curve
    ax2.plot(recall_auc, precision_auc, label="SVM (AUC)")
    ax2.plot(recall_f1, precision_f1, label="SVM (F-meas)")
    ax2.plot(recall_accel, precision_accel, label="Accel Thresh")
    ax2.set_title("Precision-Recall", fontsize=14)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if not silent:
        print(f"[{ds['name']}] : ROC and Precision-Recall plots generated successfully")
        print()


if __name__ == "__main__":
    # Main execution pipeline for Head Impact Detection using SVM
    # Dataset configuration
    dataset_path = "../dataset/head_impact_dataset.xlsx"
    dataset_name = "Head Impact Dataset"

    # Step 1: Data Preparation Pipeline
    print("=" * 60)
    print("HEAD IMPACT DETECTION USING SVM CLASSIFIER")
    print("=" * 60)
    print("Step 1: Loading and preprocessing dataset...")
    ds = load_dataset(dataset_path, dataset_name)
    ds = clean_dataset(ds)
    ds = standardize_features(ds)

    # Step 2: Statistical Analysis and Feature Assessment
    print("\nStep 2: Performing statistical analysis...")
    ds = run_wilcoxon_test(ds)
    ds = run_pca_analysis(ds)
    ds = plot_correlation_matrix(ds)

    # Step 3: Automated Feature Selection with Dual Optimization
    print("\nStep 3: Automated feature selection...")
    print("Starting Sequential Feature Selection for AUC optimization...")
    ds = select_features(ds, scoring_metric="roc_auc", n_features=8)

    print("\nStarting Sequential Feature Selection for F-measure optimization...")
    ds = select_features(ds, scoring_metric="f1", n_features=8)

    # Step 4: Model Training and Validation - AUC-Optimized Classifier
    print("\nStep 4: Training and validating AUC-optimized classifier...")
    auc_features_key = "selected_features_roc_auc"
    ds = train_and_validate(ds, feature_subset_key=auc_features_key)
    ds = test_on_independent_data(ds, feature_subset_key=auc_features_key)

    # Step 5: Model Training and Validation - F-measure-Optimized Classifier
    print("\nStep 5: Training and validating F-measure-optimized classifier...")
    f1_features_key = "selected_features_f1"
    ds = train_and_validate(ds, feature_subset_key=f1_features_key)
    ds = test_on_independent_data(ds, feature_subset_key=f1_features_key)

    # Step 6: Performance Visualization
    print("\nStep 6: Generating performance visualizations...")
    plot_roc_pr_curves(
        ds, auc_features=ds[auc_features_key], f1_features=ds[f1_features_key]
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("Check the 'output/' directory for generated visualizations.")
    print("=" * 60)
