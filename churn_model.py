"""Data generation, training, and scoring utilities for the churn demo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "late_payments"]
CATEGORICAL_FEATURES = ["contract_type", "internet_service", "paperless_billing", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


def make_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Create reproducible synthetic telecom customers for demonstration only."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, n_samples)
    monthly = np.clip(rng.normal(72, 22, n_samples), 18, 140).round(2)
    tickets = np.clip(rng.poisson(1.1, n_samples), 0, 8)
    late = np.clip(rng.poisson(0.65, n_samples), 0, 6)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[.56, .26, .18])
    internet = rng.choice(["Fiber", "DSL", "None"], n_samples, p=[.5, .35, .15])
    paperless = rng.choice(["Yes", "No"], n_samples)
    payment = rng.choice(["Electronic check", "Card", "Bank transfer", "Other"], n_samples)
    logit = (-1.15 - .035 * (tenure - 12) + .018 * (monthly - 65) + .43 * tickets + .48 * late
             + 1.0 * (contract == "Month-to-month") - .8 * (contract == "Two year")
             + .45 * (internet == "Fiber") + .35 * (payment == "Electronic check"))
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({"tenure_months": tenure, "monthly_charges": monthly,
        "support_tickets": tickets, "late_payments": late, "contract_type": contract,
        "internet_service": internet, "paperless_billing": paperless,
        "payment_method": payment, "churn": churn})


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES),
                              ("categorical", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocessor", prep), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def train_model(data: pd.DataFrame | None = None, random_state: int = 42):
    """Fit a baseline model and return (pipeline, test_features, test_labels, metrics)."""
    if data is None:
        data = make_dataset(random_state=random_state)
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if data[TARGET].nunique() != 2:
        raise ValueError("Target must contain both churn classes")
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=random_state, stratify=y)
    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
               "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0),
               "roc_auc": roc_auc_score(y_test, prob)}
    return model, X_test, y_test, metrics


def score_customer(model: Pipeline, customer: dict) -> float:
    """Return estimated churn probability for one customer (0 to 1)."""
    row = pd.DataFrame([{feature: customer[feature] for feature in FEATURES}])
    return float(model.predict_proba(row)[0, 1])


def feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return logistic-regression coefficients as directional associations."""
    prep = model.named_steps["preprocessor"]
    names = prep.get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    return pd.DataFrame({"feature": names, "coefficient": coefficients,
                         "direction": np.where(coefficients >= 0, "Higher estimated risk", "Lower estimated risk")}) \
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs()) \
        .sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
