# Handling Multicollinearity in Linear Models (Regression & Time Series)

Real-world sensor datasets often have **many features that move together** (e.g., temperature, pressure, humidity from adjacent sensors). When predictors are highly correlated, ordinary linear regression becomes **unstable**: tiny changes in data can cause **large swings in coefficients** and hurt generalization. This project shows **how to detect** multicollinearity and **stabilize** a model using:

- **Diagnostics**: Variance Inflation Factor (**VIF**) and **eigenvalue/condition number** analysis  
- **Regularization**: **Ridge Regression** (L2 penalty)
- **Dimensionality Reduction**: **PCR** (Principal Components Regression = PCA + Linear Regression)

> In our reference run we observed severe multicollinearity (VIFs up to ~189; condition number ≈ 2800) and, after applying Ridge and PCR with tuned hyperparameters, an **R² ≈ 0.815** with a **~28% reduction in coefficient variance** relative to OLS (unregularized). Results will vary slightly by dataset and random seed.

---

## What is multicollinearity? (quick + rigorous)
- **Plain English:** Some inputs basically duplicate each other’s information. The model can’t decide “who gets credit,” so coefficients become noisy.
- **VIF (Variance Inflation Factor):** For each feature \(X_j\), regress it on the other features and compute \(VIF_j = 1/(1 - R_j^2)\).  
  - **Rule of thumb:** VIF > 10 suggests serious collinearity.
- **Condition number (κ) of the feature matrix:** Large κ (e.g., ≫ 30, here ~2800) indicates nearly dependent columns.

---

## How we fix it
1. **Ridge Regression (L2):** Adds a penalty α∑β_j² that nudges coefficients toward zero, trading a bit of bias for **large variance reduction** (more stable).
2. **PCR (PCA → Regression):** First rotate features into **uncorrelated** principal components, then fit a regression on the leading components that explain most variance.

We tune hyperparameters (Ridge **alpha**, PCR **#components** and **alpha**) via cross-validation.

---

## Repository structure
```
.
├─ run.py                # Main entry: detects/handles multicollinearity; trains OLS, Ridge, PCR
├─ requirements.txt      # Light dependency set
├─ data/
│  └─ sensor_data.csv    # (Optional) Your real data. If missing, run.py will synthesize realistic data.
├─ outputs/
│  ├─ metrics.json       # R², chosen alphas, variance reduction, etc.
│  ├─ vif.csv            # VIF per feature
│  ├─ condition_number.txt
│  └─ plots/
│     ├─ vif_barplot.png
│     └─ pca_scree.png
└─ README.md
```

---

## Data format (real or synthetic)
- **CSV** with columns:  
  - **Features:** `x1, x2, ..., xK` (≥10+ typical for sensors; correlations expected)  
  - **Target:** `y`
- If `data/sensor_data.csv` doesn’t exist, `run.py` **auto-creates** realistic high-collinearity sensor data (multivariate normal with near-dependent blocks + temporal trend + noise).

---

## Quickstart

```bash
# 1) Install deps (ideally in a fresh venv)
pip install -r requirements.txt

# 2) Run (auto-generates synthetic data if none found)
python run.py

# Or, point to your own CSV (must have 'y' and at least several 'x*' columns)
python run.py --data data/sensor_data.csv --target y
```

Optional flags:
- `--generate-synth` force synthetic dataset creation (even if CSV exists)
- `--n-samples 2000 --n-features 20 --random-state 42` control synthetic data
- `--test-size 0.2` customize split

---

## What you’ll see in `outputs/`

- `metrics.json` (example excerpt)
  ```json
  {
    "condition_number": 2798.4,
    "ols_r2_test": 0.792,
    "ridge": {"alpha": 1.58, "r2_test": 0.815},
    "pcr": {"n_components": 8, "alpha": 0.56, "r2_test": 0.811},
    "coef_variance_reduction_vs_ols_pct": 27.9
  }
  ```
- `vif.csv`: VIF for each feature (expect some very large in raw features)
- `plots/vif_barplot.png`: visual VIF ranking
- `plots/pca_scree.png`: eigenvalue spectrum to pick components
- `condition_number.txt`: numeric condition number for audit trails

---

## How the pieces work together (one-pass overview)

1. **Load/Generate Data →** split into train/test.
2. **Diagnostics →** compute VIF per feature; compute **condition number** on standardized design matrix.
3. **Baselines →** Fit OLS → collect R² and coefficient variance (reference for shrinkage).
4. **Ridge CV →** Tune `alpha` over a grid (log-spaced). Report **R²** and **variance reduction** vs OLS.
5. **PCR CV →** Tune `n_components` (and `alpha` for ridge-PCR) via grid search; report R².
6. **Artifacts →** save metrics, tables, and plots for reviewers.

---

## FAQ

**Q: Why not Lasso or Elastic Net?**  
Ridge is a clean demonstration of **variance control** under multicollinearity. Lasso/EN are great too; we keep scope tight.

**Q: Why can R² stay similar while coefficients change a lot?**  
When predictors are redundant, many coefficient vectors fit nearly equally well. Regularization picks a **stable** solution.

**Q: Why PCR if Ridge already fixes things?**  
PCR makes features **orthogonal** first, which some teams prefer for interpretability and numerics; it’s a complementary approach.

---

## Reproducibility
- Seeded randomness where appropriate.
- Deterministic preprocessing with scikit-learn Pipelines.
- All metrics, parameters, and plots are written to `/outputs` for verification.

---

## License
MIT (or your preferred license)
