"""
Copied from src/overperformers.py so v2/ runs on its own.

Phase 2a: the "overperformers" model.

Idea: a college's graduation rate and earnings depend a lot on WHO it admits
(rich vs. low-income students, high vs. low SAT) and HOW MUCH money it has.
So we predict each school's outcome from those inputs alone. Then:

    residual = actual outcome - predicted outcome

A big positive residual means the school does better than expected for its
students and resources -- it "punches above its weight".
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Inputs the model is allowed to see (student body + resources, NOT outcomes)
FEATURES = [
    "pct_pell",
    "pct_first_gen",
    "admit_rate",
    "sat_avg",
    "sat_missing",       # 1 if the school doesn't report SAT (test-optional)
    "spend_per_student",
    "net_price",
]

# Outcomes we try to predict
TARGETS = ["grad_rate", "earnings_10yr"]

RANDOM_SEED = 42  # fixed so everyone gets the same numbers


def make_models():
    """Two models. Both fill missing inputs with the median first.

    - Linear regression: simple and explainable (inputs are scaled so the
      coefficients can be compared with each other).
    - Random forest: many decision trees averaged; catches non-linear patterns.
    """
    linear = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(),
    )
    forest = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1
        ),
    )
    return {"linear": linear, "forest": forest}


def evaluate(df, target):
    """Train on 80% of schools, test on the other 20%.

    Returns a list of score rows (R² and MAE per model) and the fitted models.
    """
    X, y = df[FEATURES], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    scores, fitted = [], {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores.append({
            "target": target,
            "model": name,
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        fitted[name] = model
    return scores, fitted, (X_test, y_test)


def feature_importance(fitted, X_test, y_test, target):
    """How much each input matters.

    - forest: permutation importance = how much the test R² drops when we
      shuffle that one column (bigger drop = more important).
    - linear: the scaled coefficient (sign shows the direction of the effect).
    """
    perm = permutation_importance(
        fitted["forest"], X_test, y_test,
        n_repeats=10, random_state=RANDOM_SEED, scoring="r2",
    )
    coefs = fitted["linear"][-1].coef_
    return pd.DataFrame({
        "target": target,
        "feature": FEATURES,
        "forest_importance": perm.importances_mean,
        "linear_coef": coefs,
    }).sort_values("forest_importance", ascending=False)


def predict_all_schools(df, target, model_name="forest"):
    """Predict every school using 5-fold cross-validation.

    Each school's prediction comes from a model that never saw that school,
    so its residual is fair (a model scoring its own training data would
    look too perfect and shrink every residual toward zero).
    """
    model = make_models()[model_name]
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    return cross_val_predict(model, df[FEATURES], df[target], cv=folds)


def add_residuals(df, model_name="forest"):
    """Add predicted_<target> and residual_<target> columns for each target."""
    df = df.copy()
    for target in TARGETS:
        df[f"predicted_{target}"] = predict_all_schools(df, target, model_name)
        df[f"residual_{target}"] = df[target] - df[f"predicted_{target}"]
    return df


def run(df):
    """Do everything: evaluate both models, get importances, add residuals.

    Returns (df_with_residuals, scores_table, importance_table, test_sets).
    """
    all_scores, all_importance, test_sets = [], [], {}
    for target in TARGETS:
        scores, fitted, (X_test, y_test) = evaluate(df, target)
        all_scores += scores
        all_importance.append(feature_importance(fitted, X_test, y_test, target))
        # Keep the forest's test predictions for the "actual vs predicted" chart
        test_sets[target] = pd.DataFrame({
            "actual": y_test,
            "predicted": fitted["forest"].predict(X_test),
        })

    df = add_residuals(df)
    return df, pd.DataFrame(all_scores), pd.concat(all_importance), test_sets
