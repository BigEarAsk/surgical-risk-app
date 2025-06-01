import io
import streamlit as st
from utils.Counterfact import get_res

st.title("1️⃣ Counterfact Verify")

if "models_train" not in st.session_state:
    st.warning("⬅️ Please train model on the Training page first.")
else:
    selected_models_test = st.selectbox(
        "Select one model to test or display",
        options=st.session_state.selected_models,
        # default=None  # 可自定义默认值
    )

    st.session_state.model_test = selected_models_test

    selected_features = st.multiselect(
        "Select features to be changed",
        options = st.session_state.X_train.columns,
    )

    st.session_state.selected_features_change = selected_features

    st.info("The Counterfact samples are generating...")

    counterfact_df = get_res(selected_models_test,st.session_state.models_train,
                             st.session_state.df_combined,st.session_state.continuous_features,
                             st.session_state.selected_features_change,st.session_state.target_var)

    st.dataframe(counterfact_df)

    st.info("The Counterfact samples has generated!")
