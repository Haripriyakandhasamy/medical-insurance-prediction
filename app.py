"""
Medical Insurance Cost Prediction - Streamlit App
Loads the trained pipeline (models/insurance_model.pkl) and serves
predictions + EDA + model performance pages.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pipeline_utils import add_interaction_features  # noqa: F401 (needed to unpickle model)

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="\U0001F3E5", layout="wide")

MODEL_PATH = os.path.join("models", "insurance_model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.pkl")
DATA_PATH = os.path.join("data", "medical-charges.csv")
PLOTS_DIR = os.path.join("notebooks", "plots")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    if os.path.exists(METADATA_PATH):
        return joblib.load(METADATA_PATH)
    return None


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ---------------- Sidebar Navigation ----------------
st.sidebar.title("\U0001F3E5 Insurance Cost Predictor")
page = st.sidebar.radio("Navigate", ["Home", "Data Analysis", "Prediction", "Model Performance"])

# ---------------- Home ----------------
if page == "Home":
    st.title("Medical Insurance Cost Prediction")
    st.markdown(
        """
        This app predicts **estimated medical insurance charges** for an individual
        based on their age, sex, BMI, number of children, smoking status, and region,
        using a **Linear Regression** model trained on historical insurance data.

        Use the sidebar to:
        - **Data Analysis** \u2014 explore patterns in the training data
        - **Prediction** \u2014 enter details and get an estimated charge
        - **Model Performance** \u2014 see how accurate the model is

        \u26a0\ufe0f **This tool is for educational purposes only and must not be used for
        real insurance underwriting or medical decision-making.**
        """
    )
    try:
        df_preview = load_data()
        st.subheader("Sample of the training data")
        st.dataframe(df_preview.head(10), use_container_width=True)
    except FileNotFoundError:
        st.info("Dataset not found alongside the app \u2014 prediction still works from the saved model.")

# ---------------- Data Analysis ----------------
elif page == "Data Analysis":
    st.title("\U0001F4CA Exploratory Data Analysis")
    plot_files = [
        ("01_charges_distribution.png", "Charges are right-skewed \u2014 most customers pay under $15,000, with a long tail driven mainly by smokers."),
        ("02_age_vs_charges.png", "Charges rise with age; smokers sit on a consistently higher band than non-smokers at every age."),
        ("03_bmi_vs_charges.png", "BMI barely affects non-smoker charges, but strongly increases charges for smokers \u2014 a real interaction effect."),
        ("04_smoker_vs_charges.png", "Smokers have dramatically higher charges than non-smokers, with almost no overlap."),
        ("05_region_vs_charges.png", "Charges are broadly similar across regions, with the southeast slightly higher."),
        ("06_children_vs_charges.png", "Number of children shows no strong trend with charges."),
        ("07_correlation_heatmap.png", "Among numeric features, age correlates most with charges, followed by BMI; children is nearly uncorrelated."),
    ]
    if os.path.isdir(PLOTS_DIR):
        for fname, note in plot_files:
            fpath = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True)
                st.caption(note)
                st.divider()
    else:
        st.warning("EDA plot images not found. Run `generate_eda_plots.py` and place output in `notebooks/plots/`.")

# ---------------- Prediction ----------------
elif page == "Prediction":
    st.title("\U0001F52E Predict Insurance Charges")
    st.markdown("Enter the details below and click **Predict**.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    with col2:
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    if st.button("Predict", type="primary"):
        try:
            model = load_model()
            input_df = pd.DataFrame([{
                "age": age, "sex": sex, "bmi": bmi,
                "children": children, "smoker": smoker, "region": region,
            }])
            prediction = model.predict(input_df)[0]
            prediction = max(prediction, 0)  # guard against unrealistic negative output
            st.success(f"### Estimated Insurance Charge: **${prediction:,.2f}**")
            if age < 18 or age > 100 or bmi < 10 or bmi > 60:
                st.warning("Input values are outside a typical range \u2014 treat this prediction with extra caution.")
        except FileNotFoundError:
            st.error("Model file not found. Make sure `models/insurance_model.pkl` exists.")
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

# ---------------- Model Performance ----------------
elif page == "Model Performance":
    st.title("\U0001F4C8 Model Performance")
    metadata = load_metadata()

    if metadata is not None:
        metrics = metadata.get("metrics", {})
        st.subheader("Final Model Metrics (Feature-Engineered Linear Regression)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("R\u00b2", f"{metrics.get('R2', 0):.3f}")
        m2.metric("Adjusted R\u00b2", f"{metrics.get('Adj_R2', 0):.3f}")
        m3.metric("MAE", f"${metrics.get('MAE', 0):,.0f}")
        m4.metric("MSE", f"{metrics.get('MSE', 0):,.0f}")
        m5.metric("RMSE", f"${metrics.get('RMSE', 0):,.0f}")

        st.markdown(
            """
            - **R\u00b2** \u2014 the model explains about **{:.1%}** of the variation in insurance charges.
            - **Adjusted R\u00b2** accounts for the number of features used.
            - **MAE** \u2014 predictions are off by about **${:,.0f}** on average.
            - **RMSE** \u2014 typical error, sensitive to large mistakes, is about **${:,.0f}**.
            """.format(metrics.get("R2", 0), metrics.get("MAE", 0), metrics.get("RMSE", 0))
        )

        comparison = metadata.get("comparison")
        if comparison:
            st.subheader("Baseline vs Feature-Engineered Comparison")
            st.dataframe(pd.DataFrame(comparison), use_container_width=True)
    else:
        st.warning("Model metadata not found. Metrics unavailable, but predictions will still work.")

    st.subheader("Actual vs Predicted (from training notebook)")
    plot_path = os.path.join(PLOTS_DIR, "08_actual_vs_predicted_baseline.png")
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
    else:
        st.info("Run the notebook first to generate this visualization.")

st.sidebar.markdown("---")
st.sidebar.caption("Educational project \u2014 not for real underwriting or medical use.")
