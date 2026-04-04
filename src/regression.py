import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from utils import YOUDEN_J, MIN_ERROR


def add_labels(top_k):
    df = pd.read_csv(f"csvs/results_k_{top_k}.csv")
    df["good_page_primary"] = (df["cer"] <= 0.01).astype("uint8")
    df["good_page_secondary"] = (df["cer"] <= 0.02).astype("uint8")
    return df


def train_logistic_regression_model(df, primary: bool, random_state=50):
    x = np.asarray(df["avg_bits_per_token"]).reshape(-1, 1)
    if primary:
        y = np.asarray(df["good_page_primary"])
    else:
        y = np.asarray(df["good_page_secondary"])

    # Implicitly shuffles data
    X_train, X_val, Y_train, Y_val, _, val_indices = train_test_split(
        x, y, df.index, test_size=0.20, random_state=random_state
    )

    try:
        clf = LogisticRegression(random_state=random_state).fit(X_train, Y_train)
    except Exception as e:
        raise e

    p_hat = clf.predict_proba(X_val)[:, 1]
    return p_hat, Y_val, val_indices


def compute_auc(p_hat, true_class):
    return roc_auc_score(true_class, p_hat)


def compute_roc_curve(p_hat, true_class):
    false_positive_rate, true_positive_rate, threshold = roc_curve(true_class, p_hat)
    return false_positive_rate, true_positive_rate, threshold


def plot_roc_curve(fpr, tpr, primary: bool):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker="o", markersize=3, alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if primary:
        plt.title(f"ROC Curve for 1% CER Threshold")
    else:
        plt.title(f"ROC Curve for 2% CER Threshold")
    plt.savefig("figures/roc_entropy_good_page.png")


def compute_youden_j_threshold(thresholds, fpr, tpr):
    j = tpr - fpr
    # Finds where j is maximized in thresholds
    youden_j_threshold = thresholds[np.argmax(j)]
    return youden_j_threshold


def compute_min_error_threshold(thresholds, fpr, tpr, Y_val):
    misclassification = 1 - (tpr * sum(Y_val) + (1 - fpr) * sum(1 - Y_val)) / len(Y_val)
    min_error_threshold = thresholds[np.argmin(misclassification[1:])]
    return min_error_threshold


def compute_threshold(thresholds, fpr, tpr, Y_val, threshold_type):
    if threshold_type == YOUDEN_J:
        return compute_youden_j_threshold(thresholds, fpr, tpr)

    if threshold_type == MIN_ERROR:
        return compute_min_error_threshold(thresholds, fpr, tpr, Y_val)


def compute_sensitivity(tp, fn):
    return tp / (tp + fn)


def compute_specificity(tn, fp):
    return tn / (tn + fp)

def get_misclassified_triage_decisions(top_k, use_primary, threshold_type):
     df = add_labels(top_k)
     p_hat, Y_val, val_indices = train_logistic_regression_model(df, use_primary)
     fpr, tpr, thresholds = compute_roc_curve(p_hat, Y_val)
     threshold = compute_threshold(thresholds, fpr, tpr, Y_val, threshold_type)
     Y_pred = p_hat >= threshold
     return (Y_pred == Y_val), val_indices

def main():
    # Changeable parameters
    TOP_K, USE_PRIMARY = 10, True
    if USE_PRIMARY:
        print(f"Running with 1% threshold")
    else:
        print(f"Running with 2% threshold")

    df = add_labels(TOP_K)
    p_hat, Y_val, _ = train_logistic_regression_model(df, USE_PRIMARY)
    auc = compute_auc(p_hat, Y_val)
    fpr, tpr, thresholds = compute_roc_curve(p_hat, Y_val)
    plot_roc_curve(fpr, tpr, USE_PRIMARY)

    table_data = []
    threshold_types = [YOUDEN_J, MIN_ERROR]
    for threshold_type in threshold_types:
        threshold = compute_threshold(thresholds, fpr, tpr, Y_val, threshold_type)
        Y_pred = p_hat >= threshold
        tp, fp, fn, tn = confusion_matrix(Y_val, Y_pred).ravel()
        sensitivity = compute_sensitivity(tp, fn)
        specificity = compute_specificity(tn, fp)
        table_data.append([threshold_type, threshold, sensitivity, specificity, auc])

    headers = ["Threshold Type", "Threshold", "Sensitivity", "Specificity", "AUROC"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
