# Customer Churn Prediction Dashboard

An end-to-end machine-learning portfolio project that estimates customer churn risk and explains the main drivers behind each prediction. Built with Python, pandas, scikit-learn, Plotly, and Streamlit.

> **Data note:** This starter version uses a reproducible synthetic telecom dataset. It is intended for demonstration and learning, not for real customer decisions. Replace it with an appropriately licensed, documented dataset before drawing business conclusions.

## Features
- Reproducible synthetic customer dataset with realistic churn-related signals
- Preprocessing for numeric and categorical features
- Logistic regression baseline with a held-out test set
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve
- Interactive customer risk scoring and global feature-importance view
- Unit tests for dataset generation, training, and prediction output

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```bash
pytest -q
```

## Project structure

```text
├── app.py
├── churn_model.py
├── tests/
│   └── test_churn_model.py
├── requirements.txt
└── .streamlit/config.toml
```

## Methodology
1. Generate a deterministic dataset for reproducible demos.
2. Split data into stratified train/test sets.
3. Impute numeric missing values, scale numeric features, and one-hot encode categories in a scikit-learn pipeline.
4. Train a logistic regression classifier.
5. Evaluate only on the held-out test set and present probability estimates as risk scores.

## Important limitations
- Synthetic data is not evidence of real-world model performance.
- Churn probabilities are model estimates, not guarantees.
- Feature associations are not causal explanations.
- Assess privacy, fairness, data quality, calibration, and operational impact before production use.

## License
MIT. See [LICENSE](LICENSE).