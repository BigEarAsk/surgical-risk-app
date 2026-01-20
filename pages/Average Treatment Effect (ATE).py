import io
import streamlit as st
from utils.ATE import get_ate
from common import render_lang_toggle, t

render_lang_toggle(location="sidebar")  # ✅ 这一行保证按钮

st.title("1️⃣ ATE Verify")

if "models_train" not in st.session_state:
    # st.warning("⬅️ Please train model on the Training page first.")
    st.warning(t("pdp_warning"))
else:
    selected_models_test = st.selectbox(
        # "Select one model to test or display",
        t("pdp_choose_model"),
        options=st.session_state.selected_models,
        # default=None  # 可自定义默认值
    )

    st.session_state.model_test = selected_models_test

    T_names = st.selectbox(
        # "Select T variable",
        t("ate_choose"),
        options=st.session_state.X_combined.columns,
        # default=None  # 可自定义默认值
    )

    # st.info("The ate value is calculating...")
    st.info(t("ate_calc"))

    ate_data = get_ate(st.session_state.df_combined,T_names,st.session_state.target_var,
                       st.session_state.continuous_features,st.session_state.std)
    
    st.dataframe(ate_data)
    st.info(t("ate_finish"))
    # st.info("The ate value has obtained!")
