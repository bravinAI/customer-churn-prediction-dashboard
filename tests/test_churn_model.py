import numpy as np
import pandas as pd
import pytest
from churn_model import FEATURES, make_dataset, score_customer, train_model, feature_importance


def test_dataset_is_reproducible_and_well_formed():
    a = make_dataset(300, 7)
    b = make_dataset(300, 7)
    pd.testing.assert_frame_equal(a, b)
    assert set(FEATURES + ["churn"]).issubset(a.columns)
    assert set(a.churn.unique()) == {0, 1}


def test_small_dataset_size_is_rejected():
    with pytest.raises(ValueError, match="at least 100"):
        make_dataset(99)


def test_model_metrics_and_predictions_are_valid():
    model, X_test, y_test, metrics = train_model(make_dataset(800, 10))
    assert len(X_test) == len(y_test)
    assert all(0 <= value <= 1 for value in metrics.values())
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert model.predict_proba(X_test).shape == (len(X_test), 2)


def test_customer_risk_is_probability():
    model, _, _, _ = train_model(make_dataset(800, 11))
    customer = {"tenure_months": 4, "monthly_charges": 100.0, "support_tickets": 4, "late_payments": 2,
                "contract_type": "Month-to-month", "internet_service": "Fiber",
                "paperless_billing": "Yes", "payment_method": "Electronic check"}
    risk = score_customer(model, customer)
    assert 0 <= risk <= 1


def test_feature_importance_has_directional_coefficients():
    model, _, _, _ = train_model(make_dataset(800, 12))
    importance = feature_importance(model)
    assert not importance.empty
    assert {"feature", "coefficient", "direction"}.issubset(importance.columns)
    assert np.isfinite(importance.coefficient).all()
