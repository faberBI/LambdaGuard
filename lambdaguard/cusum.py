import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lambda_detect(
    df,
    model_name,
    complexity_metric="combined",
    lambda_col="OFI_norm",
    alignment_col="A",
    smooth_window=3,
    cusum_threshold_factor=1.5,
    baseline_points=10
):
    df_model = df[df["model"] == model_name].copy()
    
    if complexity_metric == "combined":
        df_model["complexity"] = df_model["n_estimators"] * df_model["max_depth"]
    else:
        df_model["complexity"] = df_model[complexity_metric]

    df_model = df_model.sort_values("complexity").reset_index(drop=True)

    lambdas = df_model[lambda_col].values
    alignment = df_model[alignment_col].values
    complexity = df_model["complexity"].values

    # --- Normalizzazione z-score su baseline ---
    lambda_baseline = lambdas[:baseline_points]
    mu_lambda = np.mean(lambda_baseline)
    sigma_lambda = np.std(lambda_baseline) + 1e-8
    lambda_z = (lambdas - mu_lambda) / sigma_lambda

    # --- Derivate ---
    delta_lambda = np.diff(lambda_z)
    delta_lambda = pd.Series(delta_lambda).rolling(smooth_window, min_periods=1).mean().values

    delta2_lambda = np.diff(delta_lambda)
    delta2_lambda = pd.Series(delta2_lambda).rolling(smooth_window, min_periods=1).mean().values

    # --- CUSUM cumulativo ---
    mean_d2 = np.mean(delta2_lambda)
    std_d2 = np.std(delta2_lambda) + 1e-8
    centered_d2 = delta2_lambda - mean_d2

    cusum = np.zeros_like(centered_d2)
    for i in range(1, len(centered_d2)):
        cusum[i] = max(0, cusum[i-1] + centered_d2[i])

    cusum_threshold = cusum_threshold_factor * std_d2

    # --- Individuazione del change point ---
    change_index = None
    delta_alignment = np.diff(alignment)
    for i, val in enumerate(cusum):
        align_flat = delta_alignment[i] < 0.01 if i < len(delta_alignment) else False
        if val > cusum_threshold and align_flat:
            change_index = i + 2
            break

    # --- Miglior modello prima dell'overfitting basato su OFI minimo ---
    best_model_before_overfitting = None
    if change_index is not None and change_index > 0:
        pre_overfit_df = df_model.iloc[:change_index]
        best_row = pre_overfit_df.loc[pre_overfit_df[lambda_col].idxmin()]
        best_model_before_overfitting = (
            f"{best_row['model']} | {best_row['dataset']} | "
            f"min_samples_leaf={best_row['min_samples_leaf']} | "
            f"n_estimators={best_row['n_estimators']} | "
            f"max_depth={best_row['max_depth']}"
        )

    # --- Plot λ_z ---
    plt.figure(figsize=(10,5))
    plt.plot(complexity, lambda_z, '-o', label='λ_z (normalized)', color='tab:blue')
    if change_index is not None:
        plt.axvline(complexity[change_index], color='red', linestyle='--', label='Change Point')
        plt.scatter(complexity[change_index], lambda_z[change_index], color='red', s=100)
    plt.xlabel("Complexity (n_estimators*max_depth)")
    plt.ylabel("λ_z (normalized)")
    plt.title(f"Structural Overfitting Detection - {model_name}")
    plt.grid(True)
    plt.legend()
    plt.show()

    return {
        "overfitting_detected": change_index is not None,
        "change_index": change_index,
        "complexity_at_change": complexity[change_index] if change_index is not None else None,
        "lambda_z_at_change": lambda_z[change_index] if change_index is not None else None,
        "best_model_before_overfitting": best_model_before_overfitting
    }
