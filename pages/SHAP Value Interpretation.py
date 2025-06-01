import io
import streamlit as st
from utils.Shap_plot import draw
import pandas as pd

st.title("1️⃣ Shap plot")

if "models_train" not in st.session_state:
    st.warning("⬅️ Please train model on the Training page first.")
else:
    selected_models_test = st.selectbox(
        "Select one model to test or display",
        options=st.session_state.selected_models,
        # default=None  # 可自定义默认值
    )

    st.session_state.model_test = selected_models_test

    st.info("The Shap figure is drawing...")

    st.session_state.X_combined = pd.concat([st.session_state.X_train,st.session_state.X_val],ignore_index=True)
    
    fig = draw(st.session_state.model_test,st.session_state.models_train,
                   st.session_state.X_combined)

    st.pyplot(fig)

    st.info("The shap figure has finished!")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    # ✅ 下载按钮
    st.download_button(
        label="📥 Download Shap plot as PNG",
        data=buf,
        file_name="Shap_plot.png",
        mime="image/png"
    )