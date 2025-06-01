# 导入必要的库
import random
import numpy as np
import pandas as pd
# from econml.dml import DoubleMLRegressor
from doubleml import DoubleMLData, DoubleMLPLR
from econml.dml import LinearDML,NonParamDML
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(1000)
random.seed(1000)

# def get_data(path,id,t_name,y_name,sign = 0):
#     df = pd.read_excel(path,id)
#     # if sign == 1:
#     #     X = df.drop(columns=['Surgery.time','N.P.drainage','Titanium',y_name])
#     #     T = df[['Surgery.time','N.P.drainage','Titanium']]
#     # else:
#     #     X = df.drop(columns=['Surgery.time','N.P.drainage', y_name])
#     #     T = df[['Surgery.time','N.P.drainage']]
#     X = df.drop(columns=[t_name,y_name])
#     T = df[t_name]
#     # if t_name == 'Surgery.time': 
#     #     T = T / 60
#     Y = df[y_name]
#     return X,T,Y

from sklearn.decomposition import PCA

def get_data(df, t_name, y_name, sign=0, pca_flag=False, pca_var_threshold=0.95):

    # 处理变量 & 结局变量
    T = df[t_name]
    Y = df[y_name]
    X = df.drop(columns=[t_name, y_name])

    if pca_flag:
        # 直接对已标准化的协变量 X 进行 PCA 降维
        pca = PCA(n_components=pca_var_threshold)
        X_pca = pca.fit_transform(X)
        X = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    return X, T, Y

def get_ate(combined_df,T_names,Y_names,continuous_features,std):

    X,T,Y = get_data(combined_df,T_names,Y_names,0,pca_flag=True)
    # model_T = GradientBoostingRegressor(n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1)
    model_Y = GradientBoostingRegressor(n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1)
    
    if T_names in continuous_features:
        model_T = GradientBoostingRegressor(n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1)
        dml = LinearDML(model_y=model_Y, model_t=model_T)
        dml.fit(Y,T,X=X)
        ate = dml.ate(X,T0 = T,T1 = T+1/std[T_names])
        ate_ci = dml.ate_interval(X,T0 = T,T1 = T+1/std[T_names])

    else:
        model_T = GradientBoostingClassifier(n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1)
        dml = LinearDML(model_y=model_Y, model_t=model_T,discrete_treatment=True)
        dml.fit(Y,T,X=X)
        ate = dml.ate(X,T0 = 0,T1 = 1)
        ate_ci = dml.ate_interval(X,T0 = 0,T1 = 1,alpha=0.05)

    return pd.DataFrame({'Y':Y_names,'ate':ate,
                         'ate_lower':round(ate_ci[0],4),
                         'ate_upper':round(ate_ci[1],4)},index=[0])


