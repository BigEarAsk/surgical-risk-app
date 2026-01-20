import io
import streamlit as st
from utils.pr_plot import draw_pr
from common import render_lang_toggle, t

render_lang_toggle(location="sidebar")  # ✅ 这一行保证按钮
# st.title("1️⃣ PR plot")
st.title(t("pr_title"))

if "models_train" not in st.session_state:
    # st.warning("⬅️ Please train model on the Training page first.")
    st.warning(t("pdp_warning"))
else:

    selected_models_test = st.selectbox(
        # "Select one model to test or display",
        t('pdp_choose_model'),
        options=st.session_state.selected_models,
        # default=None  # 可自定义默认值
    )

    st.session_state.model_test = selected_models_test

    # st.info("The PR figure is drawing...")
    st.info(t("pr_waiting"))

    fig = draw_pr(st.session_state.model_test,st.session_state.models_train,
                   st.session_state.X_train,st.session_state.y_train,
                   st.session_state.X_val,st.session_state.y_val,
                   st.session_state.target_var)

    # st.info("The PR figure has finished!")
    st.info(t('pr_finish'))

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    # ✅ 下载按钮
    st.download_button(
        # label="📥 Download PR plot as PNG",
        label=t("pr_download"),
        data=buf,
        file_name="PR_plot.png",
        mime="image/png"
    )
