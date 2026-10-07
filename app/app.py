import joblib
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
import plotly.express as px
import shap

APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(APP_DIR, "final_model.pkl"))


@st.cache_data
def load_feature_names():
    return joblib.load(os.path.join(APP_DIR, "feature_names.pkl"))


@st.cache_data
def load_raw_data():
    df = pd.read_csv(os.path.join(APP_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


model = load_model()
feature_names = load_feature_names()
raw_df = load_raw_data()

st.title("📊 Customer Churn Prediction Dashboard")
st.caption(f"Model loaded successfully — {len(feature_names)} features")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview", "📈 EDA", "🎯 Model Performance", "🔮 Predict One Customer", "📤 Batch Upload"
])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    st.header("📋 Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(raw_df):,}")
    col2.metric("Churned", f"{(raw_df['Churn'] == 'Yes').sum():,}")
    col3.metric("Churn Rate", f"{(raw_df['Churn'] == 'Yes').mean() * 100:.1f}%")
    col4.metric("Avg. Monthly Charges", f"${raw_df['MonthlyCharges'].mean():.2f}")

    st.markdown("---")

    overview_col1, overview_col2 = st.columns([1, 1])

    with overview_col1:
        churn_counts = raw_df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig_pie = px.pie(
            churn_counts, names="Churn", values="Count",
            title="Churn Distribution", hole=0.4,
            color="Churn", color_discrete_map={"No": "#1f77b4", "Yes": "#ff4b4b"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with overview_col2:
        st.subheader("🔍 Key Insights")
        st.markdown("""
        - **Tenure is the strongest churn predictor** — customers in their first few 
          months are far more likely to leave than long-tenured customers.
        - **Contract length matters significantly** — month-to-month customers churn 
          at a much higher rate than those on one or two-year contracts.
        - **Fiber optic internet customers churn more** than DSL customers — a 
          counterintuitive finding, possibly related to price sensitivity or 
          service quality concerns in that segment.
        - **Electronic check users churn more** than customers using automatic 
          payment methods, possibly reflecting lower engagement/commitment.
        - The final model (XGBoost, F1-tuned) catches **80% of actual churners**, 
          a **39-customer improvement** in detection over the untuned baseline.
        """)

    st.markdown("---")
    st.subheader("📊 Project Summary")
    st.markdown("""
    This dashboard compares three classification models (Logistic Regression, 
    Random Forest, XGBoost) for predicting customer churn, handles class imbalance 
    through weighted training and hyperparameter tuning, and uses SHAP to explain 
    both global feature importance and individual predictions. Explore the tabs 
    above for exploratory analysis, model performance details, live single-customer 
    predictions, or batch predictions on your own data.
    """)

# ---------------- TAB 2: EDA ----------------
with tab2:
    st.header("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        contract_churn = raw_df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
        fig1 = px.bar(
            contract_churn, x="Contract", y="Count", color="Churn",
            barmode="group", title="Churn by Contract Type"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            raw_df, x="tenure", color="Churn", barmode="overlay",
            title="Churn by Tenure", nbins=30, opacity=0.7
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        internet_churn = raw_df.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
        fig3 = px.bar(
            internet_churn, x="InternetService", y="Count", color="Churn",
            barmode="group", title="Churn by Internet Service Type"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(
            raw_df, x="Churn", y="MonthlyCharges",
            title="Monthly Charges Distribution by Churn"
        )
        st.plotly_chart(fig4, use_container_width=True)

# ---------------- TAB 3: MODEL PERFORMANCE ----------------
with tab3:
    st.header("🎯 Model Performance")

    st.markdown("**Final model:** XGBoost (F1-optimized via hyperparameter tuning)")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    perf_col1.metric("Accuracy", "74.73%")
    perf_col2.metric("Precision", "51.55%")
    perf_col3.metric("Recall", "79.95%")
    perf_col4.metric("F1 Score", "62.68%")

    st.markdown("""
    **Why these numbers, and why not just accuracy:** this dataset has significant class 
    imbalance (~73% No Churn / ~27% Churn), so a naive model predicting "No Churn" for 
    everyone would already score ~73% accuracy while catching zero real churners. 
    This model was specifically tuned to optimize F1 score — balancing the ability to 
    catch real churners (Recall) against avoiding excessive false alarms (Precision).
    """)

    st.image(os.path.join(APP_DIR, "confusion_matrix_final.png"), caption="Confusion Matrix — Final Model")

# ---------------- TAB 4: SINGLE CUSTOMER PREDICTION ----------------
with tab4:
    st.header("🔮 Predict Churn for a Customer")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 1000.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        input_dict = {
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "tenure": tenure,
            "PhoneService": 1,
            "PaperlessBilling": 1,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender_Male": 1 if gender == "Male" else 0,
            "MultipleLines_No phone service": 0,
            "MultipleLines_Yes": 0,
            "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
            "InternetService_No": 1 if internet_service == "No" else 0,
            "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
            "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
            "OnlineBackup_No internet service": 0,
            "OnlineBackup_Yes": 0,
            "DeviceProtection_No internet service": 0,
            "DeviceProtection_Yes": 0,
            "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
            "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
            "StreamingTV_No internet service": 0,
            "StreamingTV_Yes": 0,
            "StreamingMovies_No internet service": 0,
            "StreamingMovies_Yes": 0,
            "Contract_One year": 1 if contract == "One year" else 0,
            "Contract_Two year": 1 if contract == "Two year" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_names]

        st.write("Input data prepared:")
        st.dataframe(input_df)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ High Churn Risk — {probability:.1%} probability")
        else:
            st.success(f"✅ Low Churn Risk — {probability:.1%} probability")

        explainer = shap.TreeExplainer(model)
        shap_values_single = explainer.shap_values(input_df)

        st.subheader("Why this prediction?")

        explanation = shap.Explanation(
            values=shap_values_single[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=input_df.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)

# ---------------- TAB 5: BATCH UPLOAD ----------------
with tab5:
    st.header("📤 Batch Prediction — Upload Your Own Customer Data")

    st.markdown("""
    Upload a CSV with the **same column structure** as the original Telco Customer Churn 
    dataset (same column names as `WA_Fn-UseC_-Telco-Customer-Churn.csv`), and this tool 
    will generate churn predictions for every customer in your file using the trained model.
    """)

    uploaded_batch = st.file_uploader("Upload customer CSV", type="csv", key="batch_upload")

    if uploaded_batch is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch)

            required_cols = [
                "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges"
            ]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()

            processed = batch_df.copy()
            processed["TotalCharges"] = pd.to_numeric(processed["TotalCharges"], errors="coerce").fillna(0)

            binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
            for col in binary_cols:
                processed[col] = processed[col].map({"Yes": 1, "No": 0})

            categorical_cols = [
                "gender", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaymentMethod"
            ]
            processed = pd.get_dummies(processed, columns=categorical_cols, drop_first=True)

            for col in feature_names:
                if col not in processed.columns:
                    processed[col] = 0
            processed = processed[feature_names]

            batch_predictions = model.predict(processed)
            batch_probabilities = model.predict_proba(processed)[:, 1]

            results = batch_df.copy()
            results["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in batch_predictions]
            results["Churn_Probability"] = (batch_probabilities * 100).round(1)

            st.success(f"Generated predictions for {len(results)} customers")
            st.dataframe(results)

            csv_output = results.to_csv(index=False)
            st.download_button(
                label="⬇️ Download predictions as CSV",
                data=csv_output,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")