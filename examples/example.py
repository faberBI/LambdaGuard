import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Import our package modules
from struct_overfitting.ofi import overfitting_index
from struct_overfitting.lambda_guard import lambda_guard_test, interpret
from struct_overfitting.cusum import detect_structural_overfitting_cusum_robust

# -----------------------------
# Generate synthetic dataset
# -----------------------------
# Using a regression dataset with 400 samples and 10 features
X, y = make_regression(n_samples=400, n_features=10, noise=0.1, random_state=42)

# -----------------------------
# Define two Gradient Boosting models
# -----------------------------
# Simple model with low complexity
simple_model = GradientBoostingRegressor(
    n_estimators=20,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
simple_model.fit(X, y)

# Complex model with high complexity (more prone to overfitting)
complex_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
complex_model.fit(X, y)

# -----------------------------
# Compute Overfitting Index (OFI)
# -----------------------------
# Measures structural overfitting: alignment, capacity, and instability
ofi_res_s = overfitting_index(simple_model, X, y)
print("Overfitting Index simple model:", ofi_res_s)

ofi_res_c = overfitting_index(complex_model, X, y)
print("Overfitting Index complex model:", ofi_res_c)

# -----------------------------
# Lambda-guard test
# -----------------------------
# Test the "leverage" of individual samples in boosting to detect overfitting spikes
lg_res_s = lambda_guard_test(simple_model, X, B=200, alpha=0.05, plot=True)
print("Lambda-guard test (simple model):", lg_res_s)
print("Interpretation:", interpret(lg_res_s))

lg_res_c = lambda_guard_test(complex_model, X, B=200, alpha=0.05, plot=True)
print("Lambda-guard test (complex model):", lg_res_c)
print("Interpretation:", interpret(lg_res_c))

# -----------------------------
# Example of CUSUM-based structural overfitting detection
# -----------------------------
# Simulate a DataFrame of Gradient Boosting models with different hyperparameters
df = pd.DataFrame([
    {"model": "GBR", "n_estimators": 50, "max_depth": 3, "A": 0.8, "OFI_norm": 0.2},
    {"model": "GBR", "n_estimators": 100, "max_depth": 5, "A": 0.85, "OFI_norm": 0.3},
    {"model": "GBR", "n_estimators": 200, "max_depth": 7, "A": 0.9, "OFI_norm": 0.5},
    {"model": "GBR", "n_estimators": 500, "max_depth": 10, "A": 0.9, "OFI_norm": 0.7},
])

# Detect potential overfitting using CUSUM method
cusum_res = detect_structural_overfitting_cusum_robust(df, model_name="GBR")
print("CUSUM Detection:", cusum_res)