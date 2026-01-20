import pandas as pd
import streamlit as st
from utils.select_var import select_feature
from common import render_lang_toggle, t

render_lang_toggle(location="sidebar")  # ✅ 这一行保证按钮
# st.title("1️⃣ Feature Selection")
st.title(t("feature_title"))

df = st.session_state.train_df

var_names = df.columns.tolist()

# target_var = st.selectbox("Select target variable", df.columns)
target_var = st.selectbox(t("feature_select"), df.columns)
X = df.drop(columns=[target_var])
y = df[target_var]

available_methods = [
        'Genetic Algorithm',
        'Lasso',
        'Random Forest-based Recursive Feature Elimination (RF-RFE)'
    ]

selected_methods = st.multiselect(
        # "Select method(s) to select features",
        t("feature_model"),
        options = available_methods,
        default = ["Genetic Algorithm"]  # 可自定义默认值
    )

results = select_feature(selected_methods,X,y,var_names)
# print(results)
if len(selected_methods) > 1:
    selected_features = []
    for res in results:
        selected_features.extend(res)
    selected_features = list(set(selected_features))
    # st.write(f'After {len(selected_methods)} methods，the selected features are：',
    #          selected_features)
    st.write(t('feature_finish'), selected_features)
    results = selected_features
    
else:
    # results = results[0]
    # st.write('The final selected features are：',results)
    st.write(t('feature_finish'), results)

X = df.drop(columns=[target_var])
X = X.loc[:,results]
y = df[target_var]

st.session_state.X_train = X
st.session_state.y_train = y
st.session_state.target_var = target_var

df2 = st.session_state.validation_df 
X_val = df2.drop(columns=[target_var])
X_val = X_val.loc[:,results]
y_val = df2[target_var]
st.session_state.X_val = X_val
st.session_state.y_val = y_val

st.session_state.X_combined = pd.concat([st.session_state.X_train,st.session_state.X_val],ignore_index=True)
st.session_state.df_combined = pd.concat([st.session_state.train_df,st.session_state.validation_df],ignore_index=True)
