import streamlit as st
import pandas as pd
from utils.model_train import train_model
from i18n import TEXT
# from utils.shap_utils import run_shap
# from utils.counterfactual import run_counterfactual
# from utils.causal_analysis import run_causal_analysis

# streamlit run /home/bigearask/python_code/文博网页设计/app.py
st.set_page_config(page_title="Surgical Risk Predictor", layout="wide")

def init_i18n():
    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    # 放在 sidebar 顶部的一键切换按钮
    toggle_label = TEXT["en"]["lang_btn_to_zh"] if st.session_state.lang == "en" else TEXT["en"]["lang_btn_to_en"]
    if st.sidebar.button(f"🌐 {toggle_label}", use_container_width=True):
        st.session_state.lang = "zh" if st.session_state.lang == "en" else "en"
        st.rerun()

def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))

init_i18n()

# st.title("🧠 Surgical Complication Risk Prediction Platform")
st.title(t("app_title"))
st.info(t("app_site"))
# st.sidebar.page_link("pages/Model_Training.py", label="Model Training")
# st.sidebar.page_link("pages/ATE.py", label="Average Treatment Effect (ATE)")
# st.sidebar.page_link("pages/Calibration_plot.py", label="Calibration Curve")
# st.sidebar.page_link("pages/Counterfact_verify.py", label="Counterfactual Validation")
# st.sidebar.page_link("pages/ROC_plot.py", label="Receiver Operating Characteristic Curve (ROC)")
# st.sidebar.page_link("pages/PR_plot.py", label="Precision-Recall Curve (PRC)")
# st.sidebar.page_link("pages/DCA_plot.py", label="Decision Curve Analysis (DCA)")
# st.sidebar.page_link("pages/Pdp_plot.py", label="Partial Dependence Plot (PDP)")
# st.sidebar.page_link("pages/Shap_plot.py", label="SHAP Value Interpretation")

# with st.expander("📘 User Guide (Click to Expand)"):
#     st.markdown("""
# ### 1.Upload Your Data File

# - **Supported format**: `.xlsx`
# - **Before uploading, please ensure the following preprocessing steps have been completed**:
#     - ✅ Missing value imputation
#     - ✅ Outlier removal
#     - ✅ Feature selection
#     - ✅ Standardization:
#         - Continuous variables: **Z-score normalization**
#         - Categorical variables: **One-hot encoding**
# ### 2. Function Execution Order

# - Please **run `Model Training` first** to build and validate baseline models.
# - After training is complete, proceed with the following steps in order:

#     1. **Performance Evaluation**  
#        - ROC Curve  
#        - Precision-Recall Curve  
#        - Calibration Curve  
#        - Decision Curve Analysis (DCA)

#     2. **Model Interpretation**  
#        - SHAP Value Interpretation  
#        - Partial Dependence Plot (PDP)

#     3. **Counterfactual Explanation**  
#        - Generate and verify diverse counterfactual scenarios

#     4. **Causal Effect Estimation**  
#        - Estimate Average Treatment Effect (ATE)
#     """)

with st.expander(t("user_guide")):
    st.markdown("\n".join([
        t("guide_title_1"),
        "",
        t("guide_supported"),
        t("guide_preprocess"),
        t("guide_p1"),
        t("guide_p2"),
        t("guide_p3"),
        t("guide_p4"),
        t("guide_p4a"),
        t("guide_p4b"),
        "",
        t("guide_title_2"),
        "",
        t("guide_order_1"),
        t("guide_order_2"),
        "",
        t("guide_block_1"),
        "",
        t("guide_block_2"),
        "",
        t("guide_block_3"),
        "",
        t("guide_block_4"),
    ]))

# uploaded_file_training = st.file_uploader("📤 Upload your training excel file (after cleaning)", type=["xlsx"])
uploaded_file_training = st.file_uploader(t("app_upload_train"), type=["xlsx"])
uploaded_file_validation = st.file_uploader(t("app_upload_valid"), type=["xlsx"])

if uploaded_file_training and uploaded_file_validation:
    df_train = pd.read_excel(uploaded_file_training)
    df_validation = pd.read_excel(uploaded_file_validation)
    st.session_state.train_df = df_train
    st.session_state.validation_df = df_validation
    # st.write("📊 Data Preview:", df.head())
    # st.success("✅ File uploaded! Please select a page from the sidebar.")
    st.success(t("app_select"))

    continue_features = st.multiselect(
        # "Select all continuous features",
        t("app_select_continue"),
        options = st.session_state.train_df.columns,
    )

    st.session_state.continuous_features = continue_features

    std = {k:0 for k in continue_features}
    for k in continue_features:
        # std[k] = st.number_input(f"请输入变量{k}的原始标准差", min_value=0.0001, format="%.4f")
        if st.session_state.lang == "en":
            std[k] = st.number_input(f"Please input the standard value of {k}", min_value=0.0001, format="%.4f")
        else:
            std[k] = st.number_input(f"请输入变量{k}的原始标准差", min_value=0.0001, format="%.4f")

    st.session_state.std = std

    # st.write("📊 Data Preview:")    
    st.write(t("app_data"))    
    st.dataframe(df_train.head())
    # st.sidebar.header("📌 Variable Selection")
    # target_var = st.sidebar.selectbox("🎯 Outcome variable (binary)", df.columns)
    # treatment_vars = st.sidebar.multiselect("🛠️ Modifiable surgical variables", df.columns.drop(target_var))

else:
    # st.info("Please upload your data to get started.")
    st.info(t("app_info"))

