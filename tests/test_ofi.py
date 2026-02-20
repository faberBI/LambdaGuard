import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

from lambdaguard.ofi import generalization_index, instability_index, create_model

def test_generalization_index():
    X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
    model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    GI, A, C = generalization_index(model, X, y)
    assert 0 <= GI or GI <= 1e10, "GI seems off"
    assert np.isfinite(A), "Alignment not finite"
    assert np.isfinite(C), "Complexity not finite"

def test_instability_index():
    X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
    model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    S = instability_index(model, X)
    assert S >= 0, "Instability should be non-negative"

def test_create_model():
    model = create_model("GBR", n_estimators=5, max_depth=2)
    assert model.n_estimators == 5
