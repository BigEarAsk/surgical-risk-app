import io
import streamlit as st
from utils.Pdp_plot import draw_pdp
import pandas as pd
from common import render_lang_toggle, t

render_lang_toggle(location="sidebar")  # ✅ 这一行保证按钮

# st.title("1️⃣ Pdp plot")
st.title(t("pdp_title"))

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

    selected_feature_test = st.selectbox(
        # "Select one feature to draw",
        t("pdp_select_feature"),
        options=st.session_state.X_train.columns,
        # default=None  # 可自定义默认值
    )

    st.session_state.feature = selected_feature_test

    # st.info("The Pdp figure is drawing...")
    st.info(t("pdp_waiting"))

    st.session_state.X_combined = pd.concat([st.session_state.X_train,st.session_state.X_val],ignore_index=True)
    
    fig = draw_pdp(st.session_state.model_test,st.session_state.models_train,
                   st.session_state.X_combined,st.session_state.feature)

    st.pyplot(fig)

    # st.info("The Pdp figure has finished!")
    st.info(t("pdp_finish"))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    # ✅ 下载按钮
    st.download_button(
        # label="📥 Download Pdp plot as PNG",
        label=t("pdp_download"),
        data=buf,
        file_name="Pdp_plot.png",
        mime="image/png"
    )
