import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats

# ===============================
# DATASET
# ===============================
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# COSTRUZIONE H SU UN FOLD
# ===============================
def build_H(model, X):
    trees = model.estimators_.ravel()
    H = np.column_stack([t.predict(X) for t in trees])
    H = H - H.mean(axis=0)  # centering sul fold
    return H

# ===============================
# METRICHE SPECTRAL WEIGHT WATCH (SPIKE VERSION)
# ===============================
def spectral_weightwatch(H_folds, spike_percent=0.15):
    lambda_max_list = []
    eff_rank_list = []
    alpha_list = []
    eigvals_all = []

    for H in H_folds:
        n, T = H.shape
        G = (H.T @ H) / n
        eigvals = np.linalg.eigvalsh(G)
        eigvals_all.append(np.sort(eigvals))
        eigvals_sorted = np.sort(eigvals)

        # λ_max
        lambda_max_list.append(eigvals_sorted[-1])

        # Effective Rank
        trace = np.sum(eigvals_sorted)
        p = eigvals_sorted / np.sum(eigvals_sorted)
        eff_rank_entropy = np.exp(-np.sum(p * np.log(p + 1e-12)))
        eff_rank_list.append(eff_rank_entropy)

        # Power-law alpha sulla coda (top 30%)
        cutoff = int(len(eigvals_sorted) * 0.7)
        tail = eigvals_sorted[cutoff:]
        tail = tail[tail > 1e-12]
        log_lambda = np.log(tail)
        hist, bin_edges = np.histogram(log_lambda, bins=20, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = hist > 0
        slope, _, _, _, _ = stats.linregress(centers[mask], np.log(hist[mask]))
        alpha_list.append(-slope)

    # λ_max e distribuzione bulk
    lambda_max_real = np.mean(lambda_max_list)

    all_eigvals = np.concatenate(eigvals_all)
    all_eigvals = np.sort(all_eigvals)

    # stima bulk edge robusta
    n_spike = max(1, int(len(all_eigvals) * spike_percent))
    bulk_eigvals = all_eigvals[:-n_spike]
    lambda_edge = lambda_edge = np.median(bulk_eigvals) + 3 * np.std(bulk_eigvals)
    sigma_bulk = np.std(bulk_eigvals)
    lambda_ratio_norm = (lambda_max_real / (lambda_edge + 1e-12)) / n_trees
    

    # effective rank e alpha medi
    eff_rank_mean = np.mean(eff_rank_list)
    alpha_mean = np.mean(alpha_list)

    return lambda_max_real, lambda_edge, sigma_bulk, eff_rank_mean, alpha_mean, eigvals_all, lambda_ratio_norm

# ===============================
# ESPERIMENTO
# ===============================
max_depth_list = [2, 3, 5, 10, 15, 20]
trees_list = [5, 10, 25, 50, 100, 300, 500]

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for max_depth in max_depth_list:
    for n_trees in trees_list:

        model = GradientBoostingRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # costruzione H su fold
        H_folds = []
        for train_idx, val_idx in kf.split(X_train):
            X_fold = X_train[train_idx]
            y_fold = y_train[train_idx]
            fold_model = GradientBoostingRegressor(
                n_estimators=n_trees,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            fold_model.fit(X_fold, y_fold)
            H_folds.append(build_H(fold_model, X_fold))

        lambda_max_real, lambda_edge, sigma_bulk, eff_rank, alpha, eigvals_all, lambda_ratio_norm = spectral_weightwatch(H_folds)

        train_rmse = mean_squared_error(y_train, model.predict(X_train))
        test_rmse = mean_squared_error(y_test, model.predict(X_test))

        # si potrebbe usare -> lambda_ratio_thresh = 1.0 + np.log10(n_trees)
        
        if (lambda_ratio_norm > 3):
            spectral_status = "⚠️ OVERFITTING DETECTED (spike)"
        else:
            spectral_status = "✅ OK"

        results.append([
            max_depth, n_trees, round(train_rmse,4), round(test_rmse,4),
            round(lambda_max_real,4), round(lambda_edge,4), round(sigma_bulk,4),
            round(eff_rank,2), round(alpha,2), spectral_status, eigvals_all, lambda_ratio_norm
        ])

# ===============================
# TABELLONE
# ===============================
df = pd.DataFrame(results, columns=[
    "max_depth", "trees", "train_rmse", "test_rmse",
    "lambda_max_real", "lambda_edge", "sigma_bulk",
    "effective_rank", "alpha", "spectral_status", "eigvals_all", "lambda_ratio_norm"
])


# ===============================
# GRAFICO AUTOVALORI CON SPIKE
# ===============================
for idx, row in df.iterrows():
    plt.figure(figsize=(8,4))
    eigvals_combined = np.concatenate(row['eigvals_all'])
    eigvals_combined = np.sort(eigvals_combined)

    plt.plot(eigvals_combined, marker='o', linestyle='', markersize=4, label='Eigenvalues')
    cutoff_idx = int(len(eigvals_combined)*(1-0.15))  # spike percent
    plt.axvline(cutoff_idx, color='gray', linestyle='--', label='Spike cutoff')
    plt.axhline(row['lambda_max_real'], color='red', linestyle='-', linewidth=2, label=f'λ_max_real={row["lambda_max_real"]:.2f}')
    plt.axhline(row['lambda_edge'], color='green', linestyle='--', linewidth=2, label=f'λ_edge={row["lambda_edge"]:.2f}')
    plt.title(f"Max Depth={row['max_depth']}, Trees={row['trees']}\nStatus: {row['spectral_status']}")
    plt.xlabel("Index of eigenvalues")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.show()

