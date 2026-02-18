# λ-Guard: Structural & Stability Overfitting Index for Gradient Boosting

## Overview

The **λ-Guard** framework is designed to detect overfitting **without using a test set**. Instead of relying on loss values, it analyzes the **geometric structure of the learned representation** and the stability of predictions under small perturbations.

It decomposes the model into two key spaces:

1. **Representation Space (Capacity)** – captures the structural complexity of the model in terms of how the input space is partitioned.  
2. **Prediction Trajectory Space (Alignment)** – measures how effectively each component (tree) contributes to the predictive target.

Each tree in Gradient Boosting partitions the input space into leaf regions. We define a **binary matrix \(Z\)**:

\[
Z_{i,j} =
\begin{cases}
1 & \text{if observation } i \text{ falls in leaf } j\\
0 & \text{otherwise}
\end{cases}
\]

- Rows correspond to observations  
- Columns correspond to leaf regions across all trees  

This matrix acts as a geometric analog to the **hat matrix (\(H\))** in linear regression: it encodes how the model “projects” the training data into its learned representation.



## Components of λ-Guard

### 1. Capacity \(C\)

The **capacity** measures the intrinsic complexity of the learned representation:

\[
C = \mathrm{Var}(Z) = \frac{1}{n} \sum_{i=1}^n \| Z_i - \bar{Z} \|^2
\]

- High \(C\) → many independent regions (complex partitioning)  
- Low \(C\) → few effective regions → simpler model  



### 2. Alignment \(A\)

Alignment quantifies how well the learned representation predicts the target:

\[
A = \mathrm{Corr}(f(X), y) \quad \text{or equivalently } A = \mathrm{Var}(f(X))
\]

- High \(A\) → each tree contributes independent functional information  
- Low \(A\) → later trees are almost linear combinations of earlier ones  



### 3. Generalization Index (GI)

The **generalization index** measures the balance between alignment and capacity:

\[
GI = \frac{A}{C}, \quad
G_{\text{norm}} = \frac{A}{A + C} \in [0,1]
\]

- \(G_{\text{norm}} \to 1\) → strong generalization  
- \(G_{\text{norm}} \to 0\) → high capacity, low alignment → potential overfitting  



### 4. Instability Index \(S\)

Measures sensitivity to small input perturbations:

\[
S = \frac{1}{n} \sum_{i=1}^{n} \frac{|f(x_i) - f(x_i + \epsilon_i)|}{\sigma_f}, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_\epsilon^2)
\]

- High \(S\) → model is unstable → small input changes lead to large prediction shifts → overfitting  

---

### 5. Overfitting Index (λ)

Combines structural complexity and stability:

\[\lambda = \frac{C}{A + C} \cdot S\]

<img src="doc/lambda_detection.png" alt="Lambda-Guard" width="600"/>
- High λ → many independent regions that **do not contribute to alignment**, plus unstable predictions → overfitting  
- Can also normalize to [0,1]:

\[\lambda_{\text{norm}} = \frac{\lambda - \min(\lambda)}{\max(\lambda) - \min(\lambda)}\]

<img src="doc/regression_gap_lambda.png"
     alt="Relationship RMSE Train/Test gap vs Lambda-Guard"
     width="600"
     style="display: block; margin: auto;" />

<img src="doc/Norm OFI California.png"
     alt="RMSE Train/Test gap vs Lambda-Guard"
     width="600"
     style="display: block; margin: auto;" />



## Geometric Interpretation

1. **Matrix \(Z\)** represents the **geometric projection** of observations into leaf regions.  
2. **Capacity \(C\)** measures the “dimensionality” of this projection.  
3. **Alignment \(A\)** captures how well this projection is aligned with the target.  
4. **Instability \(S\)** detects sensitivity to perturbations.  
5. **λ** = normalized “overfitting score” derived entirely from **training data**, without test set or OOF folds.  


<img src="doc/geometric inter.png" alt="Geometric Interpretation" width="600"/>
Geometric interpretation of Lambda-Guard. Gray squares: leaf regions, blue points: original observations, red points: instability, green arrows: alignment. High lambda occurs when capacity is high, alignment low, and instability high


Essentially, λ-Guard generalizes the concept of the hat matrix \(H\) to Gradient Boosting models: it measures **how much of the learned representation is used productively versus wasted**, geometrically.



### References / Inspirations

- λ-Guard derived from the geometric structure of the boosting hat-like matrix (Z / pseudo-residual projection)
- Gradient Boosting as a functional additive model  
- Generalization Index (GI) framework

