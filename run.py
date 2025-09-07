from __future__ import annotations
def condition_number(X: pd.DataFrame) -> float:
s = np.linalg.svd(X, full_matrices=False)[1]
return float(s[0]/s[-1])


# ---------- 3) MODELS: OLS / RidgeCV / LassoCV / PCR (CV over n_components)


def fit_pcr_cv(X, y, max_components=None):
if max_components is None:
max_components = min(X.shape)
best = (None, -1e9, None)
for n in range(1, max_components+1):
pipe = Pipeline([("pca", PCA(n_components=n)), ("lr", LinearRegression())])
pipe.fit(X, y)
r2 = r2_score(y, pipe.predict(X))
if r2 > best[1]:
best = (pipe, r2, n)
return best


# ---------- 4) METRICS helper


def metrics(y, yhat):
rmse = float(np.sqrt(mean_squared_error(y, yhat)))
r2 = float(r2_score(y, yhat))
return {"rmse": rmse, "r2": r2}


# ---------- 5) MAIN


def main(args):
X, y = make_tabular(args.n_samples, args.noise, args.seed)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)


# Diagnostics
vif = compute_vif(Xtr)
cn = condition_number(Xtr)
print("
=== DIAGNOSTICS ===")
print(vif)
print(f"Condition number: {cn:,.1f}")


# Fit models
ols = LinearRegression().fit(Xtr, ytr)
ridge = RidgeCV(alphas=np.logspace(-4,4,50)).fit(Xtr, ytr)
lasso = LassoCV(alphas=np.logspace(-4,1,50), max_iter=10000, cv=5).fit(Xtr, ytr)
pcr, pcr_r2, pcr_n = fit_pcr_cv(Xtr.values, ytr.values, max_components=min(10, Xtr.shape[1]))


# Evaluate
preds = {
"OLS": ols.predict(Xte),
"Ridge": ridge.predict(Xte),
"Lasso": lasso.predict(Xte),
"PCR": pcr.predict(Xte.values),
}
for name, yhat in preds.items():
m = metrics(yte, yhat)
print(f"{name:>6} R2={m['r2']:.3f} RMSE={m['rmse']:.3f}")


# Plot coefficient stability
fig, ax = plt.subplots()
ax.plot(ols.coef_, marker='o', label='OLS')
ax.plot(ridge.coef_, marker='o', label=f'Ridge (alpha={ridge.alpha_:.3g})')
ax.plot(lasso.coef_, marker='o', label='Lasso')
ax.set_title('Coefficient comparison')
ax.set_xlabel('Feature index')
ax.set_ylabel('Coefficient value')
ax.legend()
fig.savefig('coefficients.png', dpi=160, bbox_inches='tight')
print("Saved plot -> coefficients.png")


if __name__ == '__main__':
p = argparse.ArgumentParser()
p.add_argument('--n-samples', type=int, default=2000)
p.add_argument('--noise', type=float, default=2.0)
p.add_argument('--seed', type=int, default=42)
main(p.parse_args())
