import pandas as pd
import streamlit as st
from utils.model_train import train_model

st.title("1️⃣ Multi-Model Training")

if "train_df" not in st.session_state:
    st.warning("⬅️ Please upload data on the Home page first.")
else:
    available_models = [
    "Logistic Regression", 
    "Random Forest", 
    "Gradient Boosting", 
    "XGBoost", 
    "LightGBM",
    "Logist GAM",
    "AdaBoost",
    "CatBoost",
    "K-Nearest Neighbor",
    "Rotation Forest",
    "Guassian Process",
    "Extra Tree",
    "Decision Tree",
    "Support Vector Machine",
    "Multiply Layers Perception",
    "Bayes"
    ]

    # selected_models = st.sidebar.multiselect(
    #     "Select model(s) to train",
    #     options=available_models,
    #     default=["Random Forest", "XGBoost"]  # 可自定义默认值
    # )

    selected_models = st.multiselect(
        "Select model(s) to train",
        options = available_models,
        default = ["Random Forest", "XGBoost"]  # 可自定义默认值
    )

    st.session_state.selected_models = selected_models

    df = st.session_state.train_df
    
    target_var = st.selectbox("Select target variable", df.columns)
    X = df.drop(columns=[target_var])
    y = df[target_var]
    st.session_state.X_train = X
    st.session_state.y_train = y
    st.session_state.target_var = target_var

    df2 = st.session_state.validation_df 
    X_val = df2.drop(columns=[target_var])
    y_val = df2[target_var]
    st.session_state.X_val = X_val
    st.session_state.y_val = y_val

    st.session_state.X_combined = pd.concat([st.session_state.X_train,st.session_state.X_val],ignore_index=True)
    st.session_state.df_combined = pd.concat([st.session_state.train_df,st.session_state.validation_df],ignore_index=True)
    
    if "selected_models" in st.session_state:
        st.info("Please waiting for a moment...\nThe selected models are being trained...")
        
    results, models = train_model(X, y,X_val,y_val,st.session_state.selected_models)
    st.dataframe(results.style.highlight_max(axis=0))
    st.session_state.models_train = models
    
    if "models_train" in st.session_state:
        st.info("All selected models have been trained!")

    

    
