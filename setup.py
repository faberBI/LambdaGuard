from setuptools import setup, find_packages

setup(
    name="lambdaguard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "seaborn",
        "xgboost",
        "lightgbm",
        "catboost"
    ],
    url="https://github.com/faberBI/lambdaguard",
    author="Fabrizio Di Sciorio, PhD",
    description="Structural Overfitting detection for Gradient Boosting models",
    python_requires='>=3.8',
)
