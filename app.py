import streamlit as st
import pandas as pd
from utils.model_train import train_model
# from utils.shap_utils import run_shap
# from utils.counterfactual import run_counterfactual
# from utils.causal_analysis import run_causal_analysis

# streamlit run /home/bigearask/python_code/文博网页设计/app.py

st.set_page_config(page_title="Surgical Risk Predictor", layout="wide")
st.title("🧠 Surgical Complication Risk Prediction Platform")
# st.sidebar.page_link("pages/Model_Training.py", label="Model Training")
# st.sidebar.page_link("pages/ATE.py", label="Average Treatment Effect (ATE)")
# st.sidebar.page_link("pages/Calibration_plot.py", label="Calibration Curve")
# st.sidebar.page_link("pages/Counterfact_verify.py", label="Counterfactual Validation")
# st.sidebar.page_link("pages/ROC_plot.py", label="Receiver Operating Characteristic Curve (ROC)")
# st.sidebar.page_link("pages/PR_plot.py", label="Precision-Recall Curve (PRC)")
# st.sidebar.page_link("pages/DCA_plot.py", label="Decision Curve Analysis (DCA)")
# st.sidebar.page_link("pages/Pdp_plot.py", label="Partial Dependence Plot (PDP)")
# st.sidebar.page_link("pages/Shap_plot.py", label="SHAP Value Interpretation")

with st.expander("📘 User Guide (Click to Expand)"):
    st.markdown("""
### 1.Upload Your Data File

- **Supported format**: `.xlsx`
- **Before uploading, please ensure the following preprocessing steps have been completed**:
    - ✅ Missing value imputation
    - ✅ Outlier removal
    - ✅ Feature selection
    - ✅ Standardization:
        - Continuous variables: **Z-score normalization**
        - Categorical variables: **One-hot encoding**
### 2. Function Execution Order

- Please **run `Model Training` first** to build and validate baseline models.
- After training is complete, proceed with the following steps in order:

    1. **Performance Evaluation**  
       - ROC Curve  
       - Precision-Recall Curve  
       - Calibration Curve  
       - Decision Curve Analysis (DCA)

    2. **Model Interpretation**  
       - SHAP Value Interpretation  
       - Partial Dependence Plot (PDP)

    3. **Counterfactual Explanation**  
       - Generate and verify diverse counterfactual scenarios

    4. **Causal Effect Estimation**  
       - Estimate Average Treatment Effect (ATE)
    """)

uploaded_file_training = st.file_uploader("📤 Upload your training excel file (after cleaning)", type=["xlsx"])
uploaded_file_validation = st.file_uploader("📤 Upload your validation excel file (after cleaning)", type=["xlsx"])

if uploaded_file_training and uploaded_file_validation:
    df_train = pd.read_excel(uploaded_file_training)
    df_validation = pd.read_excel(uploaded_file_validation)
    st.session_state.train_df = df_train
    st.session_state.validation_df = df_validation
    # st.write("📊 Data Preview:", df.head())
    st.success("✅ File uploaded! Please select a page from the sidebar.")

    continue_features = st.multiselect(
        "Select all continuous features",
        options = st.session_state.train_df.columns,
    )

    st.session_state.continuous_features = continue_features

    std = {k:0 for k in continue_features}
    for k in continue_features:
        std[k] = st.number_input(f"请输入变量{k}的原始标准差", min_value=0.0001, format="%.4f")

    st.session_state.std = std

    st.write("📊 Data Preview:")    
    st.dataframe(df_train.head())
    # st.sidebar.header("📌 Variable Selection")
    # target_var = st.sidebar.selectbox("🎯 Outcome variable (binary)", df.columns)
    # treatment_vars = st.sidebar.multiselect("🛠️ Modifiable surgical variables", df.columns.drop(target_var))

    
else:
    st.info("Please upload your data to get started.")


