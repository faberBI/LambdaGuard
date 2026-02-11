# LambdaGuard

**LambdaGuard** is a framework enhanced with spectral monitoring to detect overfitting in ensemble models.

It analyzes the internal structure of the ensemble by examining the eigenvalues of the **H matrix** (tree prediction contributions) and identifies spikes, low effective rank, and unstable tail behavior. This allows early detection of overfitting beyond traditional metrics like train/test RMSE.

## Features

- Monitors key spectral metrics: `lambda_max_real`, `lambda_edge`, `effective_rank`, and tail exponent `alpha`.
- Computes a normalized spike ratio (`lambda_ratio_norm`) to account for the number of trees.
- Provides visualizations of eigenvalue distributions to detect overfitting patterns.
- Works seamlessly with tabular datasets, e.g., California Housing.

## Benefits

- Makes Gradient Boosting models more **robust** and **interpretable**.
- Detects overfitting **early**, even when standard train/test metrics look fine.
- Helps tune model complexity (depth, number of trees) with a **spectral perspective**.


