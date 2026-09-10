"""
Shared preprocessing function used inside the saved model pipeline.
Must be importable from the SAME module path at both train time and
inference time (Streamlit app), otherwise joblib/pickle cannot
reconstruct the FunctionTransformer step.
"""


def add_interaction_features(X):
    """Adds age_smoker and bmi_smoker interaction terms to a raw
    dataframe containing columns: age, sex, bmi, children, smoker, region."""
    X = X.copy()
    X["age_smoker"] = X["age"] * (X["smoker"] == "yes").astype(int)
    X["bmi_smoker"] = X["bmi"] * (X["smoker"] == "yes").astype(int)
    return X
