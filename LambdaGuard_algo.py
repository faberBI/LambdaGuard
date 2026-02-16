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
