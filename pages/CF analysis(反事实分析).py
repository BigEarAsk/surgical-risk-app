import io
import streamlit as st
from utils.Counterfact import get_res
from common import render_lang_toggle, t

render_lang_toggle(location="sidebar")  # ✅ 这一行保证按钮

# st.title("1️⃣ Counterfact Verify")
st.title(t("counterfact_title"))

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

    selected_features = st.multiselect(
        # "Select features to be changed",
        t("counterfact_change"),
        options = st.session_state.X_train.columns,
    )

    st.session_state.selected_features_change = selected_features

    # st.info("The Counterfact samples are generating...")
    st.info(t("counterfact_waiting"))

    # drop_col = list(set(st.session_state.df_combined.columns) - set(st.session_state.X_combined.columns))
    data = st.session_state.X_combined
    data[st.session_state.target_var] = st.session_state.df_combined[st.session_state.target_var]
    columns = list(st.session_state.X_train.columns)
    counterfact_df = get_res(selected_models_test,st.session_state.models_train,
                             data,columns,st.session_state.continuous_features,
                             st.session_state.selected_features_change,st.session_state.target_var)

    if counterfact_df is None or counterfact_df.empty:
        st.info(t("counterfact_none"))
        
    st.dataframe(counterfact_df)

    # st.info("The Counterfact samples has generated!")
    st.info(t("counterfact_finish"))
