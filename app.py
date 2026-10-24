"""Streamlit UI for the customer churn prediction demo."""
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from churn_model import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, feature_importance, make_dataset, score_customer, train_model

st.set_page_config(page_title="Customer Churn Lab", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction")
st.caption("A portfolio demonstration of an interpretable ML workflow. The included data is synthetic.")

with st.sidebar:
    st.header("Demo settings")
    sample_size = st.slider("Synthetic customers", 1000, 6000, 2500, step=500)
    seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42)
    st.info("Synthetic data is for demonstration only. Metrics do not represent real-world performance.")

@st.cache_data
def load_data(n, random_seed):
    return make_dataset(n_samples=n, random_state=int(random_seed))

@st.cache_resource
def fit_model(n, random_seed):
    return train_model(load_data(n, random_seed), random_state=int(random_seed))

data = load_data(sample_size, seed)
model, X_test, y_test, metrics = fit_model(sample_size, seed)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers in demo", f"{len(data):,}")
c2.metric("Synthetic churn rate", f"{data.churn.mean():.1%}")
c3.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
c4.metric("Test F1 score", f"{metrics['f1']:.3f}")

tab1, tab2, tab3 = st.tabs(["Model performance", "Score a customer", "Explore data"])
with tab1:
    st.subheader("Held-out test metrics")
    st.caption("Evaluation uses a stratified test split that was not used to fit the model.")
    st.dataframe(pd.DataFrame({"Metric": list(metrics.keys()), "Score": [round(v, 3) for v in metrics.values()]}), hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        fig_cm = px.imshow(cm, text_auto=True, x=["Predicted stay", "Predicted churn"], y=["Actual stay", "Actual churn"],
                           labels={"x":"Prediction", "y":"Actual", "color":"Customers"}, title="Confusion matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
    with right:
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        fig_roc = px.line(x=fpr, y=tpr, labels={"x":"False positive rate", "y":"True positive rate"}, title="ROC curve")
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash":"dash"})
        st.plotly_chart(fig_roc, use_container_width=True)
    st.subheader("Largest model coefficients")
    importance = feature_importance(model).head(12).sort_values("coefficient")
    st.plotly_chart(px.bar(importance, x="coefficient", y="feature", color="direction", orientation="h",
                           title="Directional associations (not causal effects)"), use_container_width=True)
with tab2:
    st.subheader("Estimate churn risk for a sample customer")
    with st.form("customer_form"):
        a, b = st.columns(2)
        with a:
            tenure = st.slider("Tenure (months)", 1, 72, 8)
            monthly = st.slider("Monthly charges", 18.0, 140.0, 85.0, step=1.0)
            tickets = st.slider("Support tickets", 0, 8, 2)
            late = st.slider("Late payments", 0, 6, 1)
        with b:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet service", ["Fiber", "DSL", "None"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Card", "Bank transfer", "Other"])
        submitted = st.form_submit_button("Estimate risk", type="primary")
    if submitted:
        customer = {"tenure_months": tenure, "monthly_charges": monthly, "support_tickets": tickets,
                    "late_payments": late, "contract_type": contract, "internet_service": internet,
                    "paperless_billing": paperless, "payment_method": payment}
        risk = score_customer(model, customer)
        st.metric("Estimated churn probability", f"{risk:.1%}")
        if risk >= .65:
            st.warning("Higher estimated risk in this demo. Treat as a prompt for review, not an automatic decision.")
        elif risk >= .35:
            st.info("Moderate estimated risk in this demo.")
        else:
            st.success("Lower estimated risk in this demo.")
with tab3:
    st.subheader("Synthetic customer data")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(data, x="tenure_months", color=data["churn"].map({0:"Stayed", 1:"Churned"}),
                                     barmode="overlay", nbins=24, labels={"color":"Outcome"}, title="Tenure distribution by outcome"), use_container_width=True)
    with right:
        rates = data.groupby("contract_type", as_index=False).agg(churn_rate=("churn", "mean"))
        st.plotly_chart(px.bar(rates, x="contract_type", y="churn_rate", title="Synthetic churn rate by contract",
                               labels={"contract_type":"Contract", "churn_rate":"Churn rate"}, text_auto=".1%"), use_container_width=True)
    st.dataframe(data.head(100), use_container_width=True)
    st.download_button("Download demo data (CSV)", data.to_csv(index=False).encode("utf-8"), file_name="synthetic_churn_data.csv", mime="text/csv")

st.divider()
st.caption("Built for learning and portfolio demonstration. Validate on licensed real data before any operational use.")