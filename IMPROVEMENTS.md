# Enhancement Roadmap — Credit Card Fraud Detection

A prioritised, honest review of what would take this project from a solid MSc dissertation to
something genuinely distinction-grade / portfolio-grade. Items are grouped by theme and tagged
with **Impact** (how much it strengthens the work) and **Effort** (rough cost). Each item points
at the concrete file/line it concerns so it is actionable.

Priority tiers:

- 🔴 **Tier 1 — Correctness & rigour.** Fix these first; they affect whether the *results are valid*.
- 🟠 **Tier 2 — Scientific depth.** These make the study more convincing and novel.
- 🟢 **Tier 3 — Engineering & polish.** These make it reproducible, professional, and deployable.

---

## 🔴 Tier 1 — Methodological correctness (do these first)

### 1.1 Decision threshold is tuned on the test set (data leakage)
**Impact: High · Effort: Low**

In `classifiers/train.py:98` and `:123`, `classifiers/ensemble.py:109`, and `classifiers/cv.py:187-189`
the optimal classification threshold is chosen to maximise F1 **on the same labels it is then
evaluated against**:

```python
best_threshold = find_best_threshold(y_test, y_proba)   # picks threshold using y_test
y_pred_adj     = (y_proba >= best_threshold)
metrics        = calculate_metrics(y_test, y_pred_adj, ...)  # evaluated on y_test
```

This optimistically biases every reported F1/precision/recall — the model has effectively "seen"
the test labels when choosing its operating point. This is the single most important thing to fix,
because an examiner who spots it will discount the headline numbers.

**Fix:** carve a *validation* split out of the training data, tune the threshold there, then apply
that fixed threshold to the untouched test set. A clean pattern:

```python
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                            stratify=y_train, random_state=25)
# fit on X_tr, choose threshold on X_val, report on X_test
```

Report the metrics both at the tuned threshold *and* at the default 0.5 for transparency.

### 1.2 Random split ignores the temporal structure of fraud
**Impact: High · Effort: Medium**

`preprocessing/preprocess.py:62` drops `Time`, and `:69` does a random stratified split. Real fraud
detection is a *forecasting* problem: you train on the past and score the future. A random split lets
the model learn from transactions that occur *after* the ones it is tested on — an unrealistic
advantage. This is a well-known critique of naïve treatments of this exact Kaggle dataset.

**Fix / dissertation angle:** add a **time-ordered split** (train on the first ~80% of `Time`, test on
the last ~20%) and compare it against the random split. Showing that performance *drops* under
temporal validation is a genuinely interesting, publishable finding and demonstrates methodological
maturity. Keep both and discuss the gap.

### 1.3 Scaling is not applied consistently inside cross-validation
**Impact: Medium · Effort: Low**

Scaling happens once in `preprocess_data` (`preprocess.py:92-107`), but `cross_validate_models`
(`cv.py`) receives the raw, unscaled `X`, `y` and only wraps resampling in a pipeline — not scaling.
So the CV results are computed on a different preprocessing regime than the final train/test results,
which makes them not directly comparable, and technically leaks scaler statistics if scaling were
added naïvely. **Fix:** put the `StandardScaler`/`RobustScaler` inside the imblearn `Pipeline` so it is
refit within every fold, and drive both the final run and CV through the *same* pipeline object.

### 1.4 Report PR-AUC (Average Precision), not just ROC-AUC
**Impact: Medium · Effort: Low**

With ~0.17% positives, ROC-AUC is misleadingly high because true negatives dominate. You already
plot PR curves — also compute **Average Precision (PR-AUC)** as a headline scalar in
`calculate_metrics` (`classifiers/utils.py:67`). It is the more honest single-number summary for this
imbalance ratio, and pairing it with MCC (which you already have) is a strong, defensible choice.

### 1.5 Single run — no measure of variability
**Impact: Medium · Effort: Medium**

Everything is one fixed-seed split. You cannot tell whether XGBoost's F1 of 0.89 vs the ensemble's
0.88 is a real difference or noise. **Fix:** report mean ± std across the CV folds (the machinery in
`cv.py` already collects per-fold metrics — surface the std), and run a **statistical significance
test** between the top models (McNemar's test on the test-set predictions, or a paired t-test across
folds). This directly addresses "is model A actually better than model B?"

---

## 🟠 Tier 2 — Scientific depth & novelty

### 2.1 Cost-sensitive evaluation (the metric that actually matters)
**Impact: High · Effort: Medium**

Fraud is an economic problem: a missed fraud (false negative) costs the transaction amount; a false
alarm (false positive) costs an investigation/customer-friction fee. F1 treats both errors equally —
banks do not. Add a **cost matrix** and report total cost / cost saved:

```
cost = FN * (mean fraud amount)  +  FP * (fixed review cost)
```

Then choose the operating threshold that *minimises expected cost* rather than maximises F1. This
reframes the whole evaluation in business terms and is a strong differentiator for the write-up. The
`Amount` column makes this directly computable.

### 2.2 Cost-sensitive learning as an alternative to resampling
**Impact: Medium · Effort: Low**

You lean entirely on SMOTE/resampling. A clean scientific comparison is **class weighting /
`scale_pos_weight`** (XGBoost), `class_weight='balanced'` (RF/MLP/LogReg) — no synthetic data at all.
Comparing "resampling vs. cost-sensitive vs. both" is a natural experimental axis and a good chapter
section. (Your CNN already uses class weights in `nn_model/model.py:169` — extend the idea to the
classical models.)

### 2.3 Add an unsupervised / anomaly-detection baseline
**Impact: High · Effort: Medium**

Every model here is supervised. Fraud detection literature strongly features **unsupervised anomaly
detection** (Isolation Forest, One-Class SVM, and an **autoencoder** reconstruction-error detector).
Adding one or two as baselines lets you argue about the supervised-vs-unsupervised trade-off, and an
autoencoder pairs naturally with your existing TensorFlow setup. This adds real novelty beyond
"compare the usual classifiers."

### 2.4 Explainability with SHAP
**Impact: High · Effort: Medium**

You produce tree feature-importance plots (`plots/feature_importance_*.png`), but fraud systems must
*justify* decisions (regulatory + trust). Add **SHAP** values for the best model (XGBoost) to show
per-prediction attributions and global feature effects. Since V1–V28 are anonymised PCA components,
frame the discussion around *which components drive fraud scores* and the interpretability limits of
PCA-obfuscated features. This is a strong, modern dissertation section.

### 2.5 Probability calibration
**Impact: Medium · Effort: Low**

Threshold-based decisions only make sense if the predicted probabilities are well-calibrated.
Add a **reliability/calibration curve** and optionally `CalibratedClassifierCV` (Platt/isotonic).
Resampling with SMOTE is known to distort output probabilities, so showing the calibration effect of
SMOTE is itself an interesting result.

### 2.6 Upgrade hyperparameter search
**Impact: Medium · Effort: Medium**

Hyperparameter tuning is currently switched **off** (`configs/config.json:13`) and uses
`RandomizedSearchCV`. Switch to **Optuna** (Bayesian/TPE) for far more sample-efficient search, log
the study, and include the optimisation history plots. Also add **LightGBM** and **CatBoost** to the
model roster — they are the current standard-bearers for tabular data and their absence is
conspicuous.

---

## 🟢 Tier 3 — Engineering, reproducibility & deployment

### 3.1 Reproducibility
**Impact: Medium · Effort: Low**

- Pin dependency versions in `requirements.txt` (currently unpinned) so results reproduce exactly.
- Set **global seeds** for `numpy`, `random`, and `tensorflow` (`tf.keras.utils.set_random_seed`) —
  right now TF runs are non-deterministic, which undermines the single-split results.
- Record the environment (`pip freeze > requirements.lock.txt`) alongside the results.

### 3.2 Replace manual config editing with a CLI
**Impact: Low · Effort: Low**

Running experiments means hand-editing `main.py` (e.g. the `use_fraction` toggle) and `config.json`.
Add `argparse` (or Hydra) so runs are `python main.py --config configs/experiment_A.json --fraction 0.1`.
This makes the experiment matrix in your dissertation trivially reproducible and self-documenting.

### 3.3 Persist and version trained models
**Impact: Medium · Effort: Low**

Only the best CNN is saved (`best_model_Neural Network.keras`). Save every fitted estimator
(`joblib.dump`) plus the fitted scalers, so evaluation/inference doesn't require retraining. Store a
small `run_metadata.json` (seed, config, git commit, metrics) per run for traceability.

### 3.4 Tests & CI
**Impact: Medium · Effort: Medium**

There are no tests. A handful of `pytest` unit tests — that `preprocess_data` doesn't leak, that
`find_best_threshold` returns a value in [0,1], that metric dicts have the expected keys — would catch
regressions and signal engineering maturity. A minimal GitHub Actions workflow running them on a tiny
synthetic sample is a nice touch.

### 3.5 A reproducible EDA / results notebook
**Impact: Medium · Effort: Low**

Add a `notebooks/` folder with (a) an **EDA notebook** (class imbalance, `Amount`/`Time`
distributions, correlation of V-features with fraud) and (b) a **results notebook** that loads
`results.log`/saved metrics and renders the comparison table and combined ROC/PR figures. This is
exactly the material that becomes dissertation figures, and it separates exploration from the
production pipeline.

### 3.6 A minimal inference / serving demo
**Impact: Medium · Effort: Medium**

To show real-world applicability, wrap the best model in a small **FastAPI** endpoint
(`POST /score` → fraud probability + decision) or a Streamlit demo. Even a single-file demo lets you
add a "Deployment considerations" chapter (latency, batch vs. real-time streaming, model monitoring,
**concept drift** as fraud patterns evolve) — a common and valued section in applied ML dissertations.

### 3.7 Housekeeping
**Impact: Low · Effort: Low**

- `results.log` is **appended** across runs, so it accumulates old results — rotate it or write
  timestamped run files to avoid confusion in the write-up.
- Fix README inconsistencies: it references `classifiers_config.json` (§Usage) but the real file is
  `configs/config.json`; the clone URL is a placeholder.
- Remove committed `__pycache__/` directories from version control and add them to `.gitignore`.
- The 150 MB `creditcard.csv` is committed to the repo — better tracked via Git LFS or `.gitignore`d
  with a download script, keeping the repo lightweight.
- De-duplicate the imports at the top of `classifiers/ensemble.py:1-14` (several are imported twice).

---

## Suggested order of attack

If you only do a few things, do them in this order for maximum marginal value:

1. **1.1 (threshold leakage)** — protects the validity of all your headline numbers.
2. **2.1 (cost-based evaluation)** — reframes the project in the terms the domain actually cares about.
3. **1.2 (temporal split)** — turns a methodological weakness into an interesting finding.
4. **2.4 (SHAP explainability)** — adds a modern, high-value chapter cheaply.
5. **1.4 + 1.5 (PR-AUC + significance testing)** — makes the comparison rigorous and defensible.

Everything else is incremental polish on top of a genuinely solid foundation.
