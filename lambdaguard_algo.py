import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes


def overfitting_index(model, X, y, noise_std=1e-3, eps=1e-8):
    """
    Computes the Overfitting Index (OFI)

    Works for:
    - Regression models
    - Binary and multiclass classification models (via predict_proba)

    Components:
    A  = Alignment (correlation between predictions and target)
    C  = Capacity (variance of predictions)
    S  = Instability (sensitivity to input noise)

    OFI = (C / (A + C)) * S
    """

    # --------------------------------------------------
    # Continuous predictions (regression or classification)
    # --------------------------------------------------
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        # Binary classification → positive class
        if proba.shape[1] == 2:
            preds = proba[:, 1]
            y_cont = y

        # Multiclass → max confidence (robust default)
        else:
            preds = np.max(proba, axis=1)
            y_cont = (np.argmax(proba, axis=1) == y).astype(float)

    else:
        # Regression
        preds = model.predict(X)
        y_cont = y

    # --------------------------------------------------
    # Alignment
    # --------------------------------------------------
    if np.std(preds) < eps or np.std(y_cont) < eps:
        A = 0.0
    else:
        A = np.corrcoef(preds, y_cont)[0, 1]

    # --------------------------------------------------
    # Capacity
    # --------------------------------------------------
    C = np.var(preds)

    # --------------------------------------------------
    # Instability
    # --------------------------------------------------
    noise = np.random.normal(0, noise_std, X.shape)

    if hasattr(model, "predict_proba"):
        proba_noisy = model.predict_proba(X + noise)
        if proba_noisy.shape[1] == 2:
            preds_noisy = proba_noisy[:, 1]
        else:
            preds_noisy = np.max(proba_noisy, axis=1)
    else:
        preds_noisy = model.predict(X + noise)

    S = np.mean(np.abs(preds - preds_noisy))
    S /= (np.std(preds) + eps)

    # --------------------------------------------------
    # Overfitting Index
    # --------------------------------------------------
    OFI = (C / (A + C + eps)) * S

    return {
        "OFI": OFI,
        "Alignment": A,
        "Capacity": C,
        "Instability": S
    }



def detect_structural_overfitting_cusum_robust(
    df,
    model_name,
    complexity_metric="combined",
    lambda_col="OFI_norm",
    alignment_col="A",
    smooth_window=3,
    cusum_threshold_factor=1.5,
):
    """
    Detect structural overfitting using second derivative + CUSUM robust detection.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment results.
    model_name : str
        Model to analyze.
    complexity_metric : str
        Column name or 'combined' (n_estimators*max_depth).
    lambda_col : str
        Column with λ values (OFI_norm).
    alignment_col : str
        Column with alignment (A).
    smooth_window : int
        Rolling window for smoothing derivatives.
    cusum_threshold_factor : float
        Multiplier for standard deviation to define CUSUM threshold.

    Returns
    -------
    dict with change point and plot.
    """
    df_model = df[df["model"] == model_name].copy()

    # Complexity axis
    if complexity_metric == "combined":
        df_model["complexity"] = df_model["n_estimators"] * df_model["max_depth"]
    else:
        df_model["complexity"] = df_model[complexity_metric]

    df_model = df_model.sort_values("complexity")
    
    lambdas = df_model[lambda_col].values
    alignment = df_model[alignment_col].values
    complexity = df_model["complexity"].values

    # First derivative Δλ
    delta_lambda = np.diff(lambdas)
    delta_lambda = pd.Series(delta_lambda).rolling(smooth_window, min_periods=1).mean().values

    # Second derivative Δ²λ
    delta2_lambda = np.diff(delta_lambda)
    delta2_lambda = pd.Series(delta2_lambda).rolling(smooth_window, min_periods=1).mean().values

    # ---------------------------------------------------
    # CUSUM cum positive
    # ---------------------------------------------------
    mean_d2 = np.mean(delta2_lambda)
    std_d2 = np.std(delta2_lambda)

    # zero-centered
    centered_d2 = delta2_lambda - mean_d2

    # CUSUM positivo
    cusum = np.zeros_like(centered_d2)
    for i in range(1, len(centered_d2)):
        cusum[i] = max(0, cusum[i-1] + centered_d2[i])

    # Threshold: mean + k*std
    cusum_threshold = cusum_threshold_factor * std_d2

    # Change point: first point where CUSUM exceeds threshold
    change_index = None
    delta_alignment = np.diff(alignment)
    for i, val in enumerate(cusum):
        align_flat = delta_alignment[i] < 0.01 if i < len(delta_alignment) else False
        if val > cusum_threshold and align_flat:
            change_index = i + 2  # shift due to double diff
            break

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(complexity, lambdas, '-o', label='λ (OFI_norm)')
    plt.plot(complexity[1:], delta_lambda, '-s', label='Δλ')
    plt.plot(complexity[2:], delta2_lambda, '-^', label='Δ²λ')
    plt.plot(complexity[2:], cusum, '-x', label='CUSUM Δ²λ', alpha=0.7)
    if change_index is not None:
        plt.axvline(complexity[change_index], color='red', linestyle='--', label='Change Point')
        plt.scatter(complexity[change_index], lambdas[change_index], color='red', s=100)
    plt.xlabel("Complexity (n_estimators*max_depth)")
    plt.ylabel("λ / Δλ / Δ²λ / CUSUM")
    plt.title(f"Structural Overfitting Detection (CUSUM) - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------------------------------
    # Output
    # ---------------------------------------------------
    if change_index is None:
        return {
            "overfitting_detected": False,
            "message": "No structural acceleration detected"
        }

    return {
        "overfitting_detected": True,
        "complexity_at_change": complexity[change_index],
        "lambda_at_change": lambdas[change_index],
        "delta_lambda_at_change": delta_lambda[change_index-1],
        "delta2_lambda_at_change": delta2_lambda[change_index-2],
        "cusum_at_change": cusum[change_index-2],
        "cusum_threshold": cusum_threshold,
        "complexity": complexity,
        "lambdas": lambdas,
        "delta_lambda": delta_lambda,
        "delta2_lambda": delta2_lambda,
        "cusum": cusum,
        "change_index": change_index
    }

result = detect_structural_overfitting_cusum_robust(
    df,
    model_name="XGB",
    complexity_metric="combined"
)


# ---------------------------------------------------
# TESTING HYP
# ---------------------------------------------------

def lambda_guard_test(model, X, B=300, alpha=0.05, plot=True):

    n = X.shape[0]
    H = boosting_leverage(model, X)

    # statistiche osservate
    T1_obs = H.sum() / n
    T2_obs = H.max() / H.mean()

    T1_boot = np.zeros(B)
    T2_boot = np.zeros(B)

    for b in range(B):
        idx = np.random.choice(n, n, replace=True)
        Hb = boosting_leverage(model, X[idx])
        T1_boot[b] = Hb.sum() / n
        T2_boot[b] = Hb.max() / Hb.mean()

    # quantili critici (one-sided)
    q1 = np.quantile(T1_boot, 1 - alpha)
    q2 = np.quantile(T2_boot, 1 - alpha)

    # p-values
    p1 = np.mean(T1_boot >= T1_obs)
    p2 = np.mean(T2_boot >= T2_obs)

    reject = (p1 < alpha) or (p2 < alpha)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # ---- T1: df_ratio ----
        ax = axes[0]
        ax.hist(T1_boot, bins=30, density=True, alpha=0.7)
        ax.axvline(T1_obs, color="black", linewidth=2, label="Observed")
        ax.axvline(q1, color="red", linestyle="--", label="Critical (1-α)")

        ax.axvspan(q1, T1_boot.max(),
                   alpha=0.25, label="Reject H₀ region")

        ax.set_title("T1: Effective DoF ratio")
        ax.set_xlabel("df_ratio")
        ax.set_ylabel("Density")
        ax.legend()

        # ---- T2: peak_ratio ----
        ax = axes[1]
        ax.hist(T2_boot, bins=30, density=True, alpha=0.7)
        ax.axvline(T2_obs, color="black", linewidth=2, label="Observed")
        ax.axvline(q2, color="red", linestyle="--", label="Critical (1-α)")

        ax.axvspan(q2, T2_boot.max(),
                   alpha=0.25, label="Reject H₀ region")

        ax.set_title("T2: Peak leverage ratio")
        ax.set_xlabel("max(Hᵢᵢ) / mean(H)")
        ax.set_ylabel("Density")
        ax.legend()

        plt.tight_layout()
        plt.show()

    return {
        "T1_df_ratio": T1_obs,
        "critical_df_ratio": q1,
        "p_df_ratio": p1,
        "T2_peak_ratio": T2_obs,
        "critical_peak_ratio": q2,
        "p_peak_ratio": p2,
        "reject_H0": reject
    }


def boosting_leverage(model, X):
    """
    H_ii ≈ self influence del punto i
    basato sulle foglie degli alberi del boosting
    """
    n = X.shape[0]
    influence = np.zeros(n)

    for est in model.estimators_.ravel():
        leaf_id = est.apply(X)

        unique, counts = np.unique(leaf_id, return_counts=True)
        leaf_sizes = dict(zip(unique, counts))

        lr = model.learning_rate

        for i in range(n):
            influence[i] += lr / leaf_sizes[leaf_id[i]]

    return influence


def interpret(res):
    if not res["reject_H0"]:
        return "✔ REGIME STABILE / GENERALIZZANTE"
    if res["p_df_ratio"] < 0.05 and res["p_peak_ratio"] < 0.05:
        return "✖ REGIME INTERPOLANTE (OVERFITTING FORTE)"
    if res["p_df_ratio"] < 0.05:
        return "✖ COMPLESSITÀ GLOBALE ECCESSIVA"
    return "✖ (LEVERAGE SPIKES)"


