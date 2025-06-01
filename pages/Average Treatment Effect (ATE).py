import io
import streamlit as st
from utils.ATE import get_ate

st.title("1️⃣ ATE Verify")

if "models_train" not in st.session_state:
    st.warning("⬅️ Please train model on the Training page first.")
else:
    selected_models_test = st.selectbox(
        "Select one model to test or display",
        options=st.session_state.selected_models,
        # default=None  # 可自定义默认值
    )

    st.session_state.model_test = selected_models_test

    T_names = st.selectbox(
        "Select T variable",
        options=st.session_state.X_combined.columns,
        # default=None  # 可自定义默认值
    )

    st.info("The ate value is calculating...")

    ate_data = get_ate(st.session_state.df_combined,T_names,st.session_state.target_var,
                       st.session_state.continuous_features,st.session_state.std)
    
    st.dataframe(ate_data)
    st.info("The ate value has obtained!")
