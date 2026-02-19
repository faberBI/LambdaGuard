<p align="center">🛡️ λ-Guard

Overfitting detection for Gradient Boosting — no validation set required

<i>Understand when boosting stops learning signal and starts memorizing structure.</i>

</p>---

❓ Why λ-Guard

In Gradient Boosting, overfitting usually appears after the real problem has already started.

Before validation error increases, the model is already:

- splitting the feature space into extremely small regions
- fitting leaves supported by very few observations
- becoming sensitive to tiny perturbations

The model is not improving prediction anymore.

It is learning the shape of the training dataset.

λ-Guard detects that moment.

---

🧠 The intuition

A boosting model learns two different things at the same time:

Component| What it does
Geometry| partitions the feature space
Predictor| assigns values to each region

Overfitting happens when:

«the geometry keeps growing but the predictor stops gaining real information.»

So λ-Guard measures three signals:

- 📦 capacity → how complex the partition is
- 🎯 alignment → how much signal is extracted
- 🌊 stability → how fragile predictions are

---

🧩 Representation (the key object)

Every tree divides the feature space into leaves.

We record where each observation falls and build a binary matrix Z:

Z(i,j) = 1  if sample i falls inside leaf j
Z(i,j) = 0  otherwise

Rows → observations
Columns → all leaves across all trees

Think of Z as the representation learned by the ensemble.

Linear regression → hat matrix H
Boosting → representation matrix Z

---

📦 Capacity — structural complexity

C = Var(Z)

What it means:

- low C → the model uses few effective regions
- high C → the model fragments the space

When boosting keeps adding trees late in training, C grows fast.

---

🎯 Alignment — useful information

A = Corr(f(X), y)

(or equivalently the variance of predictions)

- high A → trees add real predictive signal
- low A → trees mostly refine boundaries

Important behavior:

«After some number of trees, alignment saturates.»

Boosting continues building structure even when prediction stops improving.

---

🌊 Instability — sensitivity to perturbations

We slightly perturb inputs:

x' = x + ε
ε ~ Normal(0, σ²)

and measure prediction change:

S = average |f(x) − f(x')|  /  prediction_std

- low S → smooth model
- high S → brittle model

This is the first thing that explodes during overfitting.

---

🔥 The Overfitting Index

λ = ( C / (A + C) ) × S

Interpretation:

Situation| λ
compact structure + stable predictions| low
many regions + weak signal| high
unstable predictions| very high

λ measures:

«how much structural complexity is wasted.»

(You can normalize λ to [0,1] for comparisons.)

---

🧪 Structural Overfitting Test

We can also check if specific training points dominate the model.

Approximate leverage:

H_ii ≈ Σ_trees (learning_rate / leaf_size)

This behaves like regression leverage.

We compute:

T1 = mean(H_ii)        # global complexity
T2 = max(H_ii)/mean(H_ii)   # local memorization

Bootstrap procedure

repeat B times:
    resample training data
    recompute T1, T2

p-values:

p1 = P(T1_boot ≥ T1_obs)
p2 = P(T2_boot ≥ T2_obs)

Reject structural stability if:

p1 < α  OR  p2 < α

---

📊 What λ-Guard distinguishes

Regime| Meaning
✅ Stable| smooth generalization
📈 Global overfitting| too many effective parameters
⚠️ Local memorization| few points dominate
💥 Extreme| interpolation behavior

---

🧭 When to use

- monitoring boosting while trees are added
- hyperparameter tuning
- small datasets (no validation split)
- diagnosing late-stage performance collapse

---

🧾 Conceptual summary

Z  → learned representation
C  → structural dimensionality
A  → extracted signal
S  → smoothness
λ  → structural overfitting

Overfitting = structure grows faster than information.

---

📜 License

MIT (edit as needed)