# ============================================================
# FULL EXPERIMENT: LAMBDA GUARD 
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes


# ============================================================
# GENERALIZATION COMPONENTS
# ============================================================

def generalization_index(model, X, y):
    """
    A = Alignment with target (train R correlation)
    C = Capacity (variance of predictions)
    """
    preds = model.predict(X)

    # Alignment = correlation with target
    A = np.corrcoef(preds, y)[0, 1]

    # Capacity = variance of predictions
    C = np.var(preds)

    # Avoid division problems
    GI = A / C if C > 0 else 0

    return GI, A, C


def instability_index(model, X, noise_std=1e-3):
    """
    Measures prediction sensitivity to small input perturbations
    """
    preds_clean = model.predict(X)
    noise = np.random.normal(0, noise_std, X.shape)
    X_noisy = X + noise
    preds_noisy = model.predict(X_noisy)

    instability = np.mean(np.abs(preds_clean - preds_noisy))
    instability /= (np.std(preds_clean) + 1e-8)

    return instability


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(X, y, dataset_name):

    print("\n" + "="*70)
    print(f"DATASET: {dataset_name}")
    print("="*70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    n_estimators_list = [20, 100, 500]
    max_depth_list = [3, 7, 15]
    min_samples_leaf_list = [2, 5, 10, 15]

    results = []

    for sl in min_samples_leaf_list:
        for n_est in n_estimators_list:
            for depth in max_depth_list:

                model = GradientBoostingRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_leaf=sl,
                    random_state=42
                )

                model.fit(X_train, y_train)

                # ---- Structural Components ----
                GI, A, C = generalization_index(model, X_train, y_train)
                G_norm = A / (A + C)

                # ---- Stability ----
                S = instability_index(model, X_train)

                # ---- Overfitting Index ----
                OFI = (C / (A + C)) * S

                results.append({
                    "dataset": dataset_name,
                    "min_samples_leaf": sl,
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "A": A,
                    "C": C,
                    "GI": GI,
                    "G_norm": G_norm,
                    "Instability": S,
                    "OFI": OFI
                })

    df = pd.DataFrame(results)

    # ---- Normalize OFI ----
    OFI_min = df["OFI"].min()
    OFI_max = df["OFI"].max()
    df["OFI_norm"] = (df["OFI"] - OFI_min) / (OFI_max - OFI_min)

    # ---- Compute RMSE on train/test ----
    for idx, row in df.iterrows():
        model = GradientBoostingRegressor(
            n_estimators=int(row["n_estimators"]),
            max_depth=int(row["max_depth"]),
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=int(row["min_samples_leaf"]),
            random_state=42
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        gap = rmse_test - rmse_train

        df.at[idx, "Train_RMSE"] = rmse_train
        df.at[idx, "Test_RMSE"] = rmse_test
        df.at[idx, "Gap"] = gap

    print("\nTop 5 Lowest Test RMSE:")
    print(df.sort_values("Test_RMSE").head())

    return df


# ============================================================
# PLOTTING
# ============================================================

def plot_all(df, dataset_name):

    # G_norm vs Gap
    plt.figure(figsize=(6,5))
    sns.regplot(data=df, x="G_norm", y="Gap")
    plt.title(f"{dataset_name} - G_norm vs Gap")
    plt.grid(True)
    plt.show()

    # OFI normalized vs Gap
    plt.figure(figsize=(6,5))
    sns.regplot(data=df, x="OFI_norm", y="Gap")
    plt.title(f"{dataset_name} - Normalized OFI vs Gap")
    plt.grid(True)
    plt.show()

    # Heatmap OFI_norm
    pivot_ofi = df.pivot_table(
        values="OFI_norm",
        index="max_depth",
        columns="n_estimators",
        aggfunc="mean"
    )

    plt.figure(figsize=(6,5))
    sns.heatmap(pivot_ofi, annot=True, fmt=".3f", cmap="Purples")
    plt.title(f"{dataset_name} - Normalized OFI Heatmap")
    plt.show()

    # Heatmap Gap
    pivot_gap = df.pivot_table(
        values="Gap",
        index="max_depth",
        columns="n_estimators",
        aggfunc="mean"
    )

    plt.figure(figsize=(6,5))
    sns.heatmap(pivot_gap, annot=True, fmt=".3f", cmap="Reds")
    plt.title(f"{dataset_name} - Gap Heatmap")
    plt.show()


# ============================================================
# RUN
# ============================================================

diab = load_diabetes()
df_diab = run_experiment(diab.data, diab.target, "Diabetes")
plot_all(df_diab, "Diabetes")


import numpy as np
import pandas as pd

df = pd.read_excel('/content/results_model_lambda_guard.xlsx')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    # CUSUM cumulativo positivo
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


def gap_lambdaguard_test(df):

  X = df['OFI_norm']
  y = df['Gap']

  # Aggiungo intercept
  X_const = sm.add_constant(X)

  # Fit regressione lineare
  model = sm.OLS(y, X_const).fit()

  # Risultati
  print(model.summary())

  # Coefficiente beta
  beta = model.params['OFI_norm']
  print(f"Coefficiente beta: {beta:.4f}")

  # p-value del test H0: beta=0
  pvalue = model.pvalues['OFI_norm']
  print(f"P-value test beta=0: {pvalue:.4f}")

  # Plot
  plt.figure(figsize=(8,6))
  plt.scatter(df['OFI_norm'], df_model['Gap'], alpha=0.6)
  plt.plot(df['OFI_norm'], model.predict(X_const), color='red', linewidth=2)
  plt.xlabel('Lambda')
  plt.ylabel('RMSE gap (test - train)')
  plt.title(f'Regression Gap RMSE vs Lambda - {df_model["model"].iloc[0]}')
  plt.grid(True)
  plt.show()


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



