#!/usr/bin/env python3
"""
Handling Multicollinearity: Diagnostics (VIF, condition number) + Ridge + PCR

- Loads a CSV with features x1..xK and target y
- If data is missing (or --generate-synth), generates realistic high-collinearity sensor data
- Computes VIF, condition number, trains OLS, RidgeCV, and PCR (with CV)
- Writes metrics/tables/plots to ./outputs
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional: VIF via statsmodels if available; otherwise compute via R^2 trick
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as sm_vif
    _HAS_SM = True
except Exception:
    _HAS_SM = False


# --------------------------- Utilities ---------------------------------

def ensure_dirs():
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)5s | %(message)s",
        datefmt="%H:%M:%S",
    )


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def condition_number(X_std: np.ndarray) -> float:
    """Compute matrix condition number on standardized features (unit variance)."""
    u, s, v = np.linalg.svd(X_std, full_matrices=False)
    return float((s.max() / max(s.min(), 1e-15)))


def compute_vif_df(X: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF per column; returns DataFrame with ['feature', 'vif'] sorted desc."""
    X_ = X.copy()
    X_ = X_.assign(constant=1.0)  # intercept
    cols = list(X.columns)

    vifs = []
    if _HAS_SM:
        for i, col in enumerate(cols):
            v = sm_vif(X_.values, i)  # ignore the added constant at the end
            vifs.append((col, float(v)))
    else:
        # Fallback (no statsmodels): VIF_j = 1/(1 - R_j^2),
        # where R_j^2 is from regressing x_j on X_{-j}
        from sklearn.linear_model import LinearRegression
        for j, col in enumerate(cols):
            y = X[[col]].values.ravel()
            X_minus = X.drop(columns=[col]).values
            lr = LinearRegression().fit(X_minus, y)
            r2 = lr.score(X_minus, y)
            vif = 1.0 / max(1.0 - r2, 1e-12)
            vifs.append((col, float(vif)))

    vif_df = pd.DataFrame(vifs, columns=["feature", "vif"]).sort_values("vif", ascending=False)
    return vif_df


def coef_variance(coef: np.ndarray) -> float:
    """A simple, model-agnostic proxy: empirical variance of coefficients."""
    c = np.asarray(coef).ravel()
    if c.size == 0:
        return 0.0
    return float(np.var(c))


# ------------------------ Synthetic Data --------------------------------

def make_synthetic_sensor_data(n_samples=1500, n_features=18, random_state=42):
    """
    Create block-correlated features mimicking sensor redundancy.
    - 3 blocks of 6 features each with high within-block correlation (~0.92-0.97)
    - Add a mild temporal component and noise
    - Construct target as a sparse-ish linear combo + noise
    Returns DataFrame with columns x1..xK and y.
    """
    rng = np.random.default_rng(random_state)

    def block_cov(k, base_corr):
        M = np.full((k, k), base_corr)
        np.fill_diagonal(M, 1.0)
        return M

    k = n_features
    blocks = []
    for base_corr in [0.92, 0.95, 0.97]:
        blocks.append(block_cov(k // 3, base_corr))
    rem = k - sum(b.shape[0] for b in blocks)
    if rem > 0:
        blocks.append(np.eye(rem) * 1.0)

    cov = np.block([
        [blocks[i] if i == j else np.zeros((blocks[i].shape[0], blocks[j].shape[0]))
         for j in range(len(blocks))]
        for i in range(len(blocks))
    ])

    mean = np.zeros(k)
    X = rng.multivariate_normal(mean, cov, size=n_samples)

    t = np.linspace(0, 1, n_samples)
    drift = 0.3 * t[:, None] * rng.normal(0.8, 0.05, size=(1, k))
    X = X + drift

    true_coef = np.zeros(k)
    true_idx = rng.choice(k, size=max(3, k // 6), replace=False)
    true_coef[true_idx] = rng.normal(2.0, 0.4, size=true_idx.size) * rng.choice([-1, 1], size=true_idx.size)
    y = X @ true_coef + 0.8 * rng.normal(0, 1, size=n_samples)

    cols = [f"x{i+1}" for i in range(k)]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y

    return df


# ---------------------------- Main --------------------------------------

def main():
    setup_logging()
    ensure_dirs()

    parser = argparse.ArgumentParser(description="Handle Multicollinearity: Diagnostics + Ridge + PCR")
    parser.add_argument("--data", type=str, default="data/sensor_data.csv",
                        help="Path to CSV with x* features and target column (--target).")
    parser.add_argument("--target", type=str, default="y", help="Target column name.")
    parser.add_argument("--generate-synth", action="store_true",
                        help="Force generation of synthetic data into --data path.")
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--n-features", type=int, default=18)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)

    if args.generate_synth or (not data_path.exists()):
        logging.info("Generating synthetic sensor dataset with strong multicollinearity...")
        df = make_synthetic_sensor_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            random_state=args.random_state,
        )
        df.to_csv(data_path, index=False)
        logging.info(f"Synthetic data written to {data_path.resolve()}")
    else:
        logging.info(f"Loading data from {data_path.resolve()}")
        df = pd.read_csv(data_path)

    assert args.target in df.columns, f"Target column '{args.target}' not found in data."
    feature_cols = [c for c in df.columns if c != args.target]
    X = df[feature_cols].copy()
    y = df[args.target].values

    logging.info("Computing VIF and condition number...")
    vif_df = compute_vif_df(X)
    vif_df.to_csv("outputs/vif.csv", index=False)

    X_std = StandardScaler().fit_transform(X.values)
    kappa = condition_number(X_std)
    with open("outputs/condition_number.txt", "w") as f:
        f.write(f"{kappa:.4f}\n")
    logging.info(f"Condition number (std. features): {kappa:.1f}")

    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 6))
        top = vif_df.head(min(25, len(vif_df)))
        plt.barh(top["feature"][::-1], top["vif"][::-1])
        plt.xlabel("VIF (Variance Inflation Factor)")
        plt.title("Top VIFs (higher = more collinear)")
        plt.tight_layout()
        fig.savefig("outputs/plots/vif_barplot.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        logging.warning(f"Could not create VIF plot: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=args.test_size, random_state=args.random_state
    )

    scaler = StandardScaler()

    logging.info("Fitting baseline OLS...")
    ols = Pipeline(steps=[
        ("scaler", scaler),
        ("lr", LinearRegression())
    ])
    ols.fit(X_train, y_train)
    y_pred_test = ols.predict(X_test)
    ols_r2_test = r2_score(y_test, y_pred_test)
    ols_coef = ols.named_steps["lr"].coef_
    ols_coef_var = coef_variance(ols_coef)

    logging.info("Fitting RidgeCV...")
    alphas = np.logspace(-6, 3, 30)
    ridgecv = Pipeline(steps=[
        ("scaler", scaler),
        ("ridge", RidgeCV(alphas=alphas, store_cv_values=False))
    ])
    ridgecv.fit(X_train, y_train)
    ridge_alpha = float(ridgecv.named_steps["ridge"].alpha_)
    ridge_coef = ridgecv.named_steps["ridge"].coef_
    ridge_r2_test = r2_score(y_test, ridgecv.predict(X_test))
    ridge_coef_var = coef_variance(ridge_coef)
    variance_reduction_pct = 100.0 * (ols_coef_var - ridge_coef_var) / max(ols_coef_var, 1e-12)

    logging.info("Fitting PCR (PCA + Ridge) with GridSearchCV...")
    pcr_pipe = Pipeline(steps=[
        ("scaler", scaler),
        ("pca", PCA(svd_solver="full", random_state=args.random_state)),
        ("ridge", Ridge())
    ])
    max_comp = min(30, X_train.shape[1])
    pcr_param_grid = {
        "pca__n_components": list(range(3, max_comp + 1)),
        "ridge__alpha": np.logspace(-3, 2, 10),
    }
    pcr_cv = GridSearchCV(
        pcr_pipe, pcr_param_grid,
        scoring="r2", cv=5, n_jobs=-1, verbose=0
    )
    pcr_cv.fit(X_train, y_train)
    pcr_best = pcr_cv.best_estimator_
    pcr_best_params = pcr_cv.best_params_
    pcr_r2_test = r2_score(y_test, pcr_best.predict(X_test))

    try:
        pca_all = PCA(svd_solver="full", random_state=args.random_state).fit(scaler.fit_transform(X.values))
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(pca_all.explained_variance_) + 1), pca_all.explained_variance_, marker="o")
        plt.xlabel("Principal Component")
        plt.ylabel("Eigenvalue (variance explained)")
        plt.title("PCA Scree Plot")
        plt.tight_layout()
        fig.savefig("outputs/plots/pca_scree.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        logging.warning(f"Could not create scree plot: {e}")

    metrics = {
        "condition_number": round(kappa, 3),
        "ols_r2_test": round(ols_r2_test, 4),
        "ridge": {
            "alpha": round(ridge_alpha, 5),
            "r2_test": round(ridge_r2_test, 4),
        },
        "pcr": {
            "n_components": int(pcr_best_params["pca__n_components"]),
            "alpha": float(pcr_best_params["ridge__alpha"]),
            "r2_test": round(pcr_r2_test, 4),
        },
        "coef_variance_reduction_vs_ols_pct": round(variance_reduction_pct, 1),
    }
    save_json(metrics, "outputs/metrics.json")

    logging.info("Done.")
    logging.info(f"Condition number: {metrics['condition_number']}")
    logging.info(f"Test R² — OLS: {metrics['ols_r2_test']} | Ridge: {metrics['ridge']['r2_test']} | PCR: {metrics['pcr']['r2_test']}")
    logging.info(f"Ridge alpha: {metrics['ridge']['alpha']} | PCR best: {metrics['pcr']}")
    logging.info(f"Coefficient variance reduction vs OLS: {metrics['coef_variance_reduction_vs_ols_pct']}%")

if __name__ == "__main__":
    main()
