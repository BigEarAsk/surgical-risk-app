import io
import streamlit as st
from utils.pr_plot import draw_pr

st.title("1️⃣ PR plot")

if "models_train" not in st.session_state:
    st.warning("⬅️ Please train model on the Training page first.")
else:

    selected_models_test = st.selectbox(
        "Select one model to test or display",
        options=st.session_state.selected_models,
        # default=None  # 可自定义默认值
    )

    st.session_state.model_test = selected_models_test

    st.info("The PR figure is drawing...")

    fig = draw_pr(st.session_state.model_test,st.session_state.models_train,
                   st.session_state.X_train,st.session_state.y_train,
                   st.session_state.X_val,st.session_state.y_val,
                   st.session_state.target_var)

    st.info("The PR figure has finished!")

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    # ✅ 下载按钮
    st.download_button(
        label="📥 Download PR plot as PNG",
        data=buf,
        file_name="PR_plot.png",
        mime="image/png"
    )