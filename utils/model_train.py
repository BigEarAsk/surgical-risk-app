from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .Grid_optim import search
import xlrd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn import svm, tree
import random
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from ngboost import NGBClassifier
from sklearn.linear_model import LogisticRegression
from pygam import LogisticGAM, s
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils import resample
from scipy.stats import bootstrap
from .bootstrap import calculate_metrics,bootstrap_metrics,bootstrap_metrics_bca
# import streamlit as st


# file = xlrd.open_workbook("递归消除特征/Derivation cohort (2).xlsx")
random.seed(100)
np.random.seed(100)
model_names = ['gam','logit','ada','cat','gbdt','knn','lgb', 
                  'rotation','xgboost', 'rf','guassin_cls','extra_tree',
                  'dt','svm','mlp','bayes']
model2id = {
    "Logistic Regression":1,
    "Random Forest":9,
    "Gradient Boosting":4,
    "XGBoost":8,
    "LightGBM":6,
    "AdaBoost":2,
    "CatBoost":3,
    "K-Nearest Neighbor":5,
    "Rotation Forest":7,
    "Guassian Process":10,
    "Extra Tree":11,
    "Logist GAM":0,
    "Decision Tree":12,
    "Support Vector Machine":13,
    "Multiply Layers Perception":14,
    "Bayes":15
}

model_name_transfer = {
    'gam':"Logist GAM",
    'logit':"Logistic Regression",
    'ada':"AdaBoost",
    'cat':"CatBoost",
    'gbdt':"Gradient Boosting", 
    'knn':"K-Nearest Neighbor",
    'lgb':"LightGBM",
    'rotation':"Rotation Forest",
    'xgboost':"XGBoost",
    'rf':"Random Forest", 
    'guassin_cls':"Guassian Process",
    'extra_tree':"Extra Tree",
    'dt':"Decision Tree",
    'svm':"Support Vector Machine",
    'bayes':"Bayes",
    "mlp":"Multiply Layers Perception"
}

# set hyperparameters search value range of all models:
svm_param_dict = {
        'C':[3.,7.,10.],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree':[3,5],
    }

decision_tree_param_dict = {
        'criterion':["gini", "entropy", "log_loss"],
        'max_depth':[None,5,10,15,20],
        'min_samples_leaf':[1,2,3,4]
    }

extra_tree_param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20],  
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt', 'log2']  
}

guassin_cls_param_grid = {
    'kernel': [RBF(), None],  #
    'n_restarts_optimizer': [0, 1, 2],  
}

rf_param_grid = {  
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20], 
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt', 'log2']  
}

elastic_net_param_grid = {
    'alpha': [0.01, 0.1, 1.0],  
    'l1_ratio': [0.1, 0.5, 0.9],  
    'max_iter': [1000, 2000, 3000]  
}

bayes_param_grid = {  
    'selector__k': [5, 10, 15]  
}

xgboost_param_grid = {
    'n_estimators': [50, 100, 200], 
    'learning_rate': [0.01, 0.1, 0.5],  
    'max_depth': [3, 5, 7]  
}

rotation_forest_param_grid = {
    'pca__n_components': [3,5,7],
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [3, 5, 7]
}

ngboost_param_grid = {
    'n_estimators': [50, 100, 200],  
    'learning_rate': [0.01, 0.1, 0.5],  
    'minibatch_frac': [0.1, 0.3, 0.5]  
}

lgb_param_grid = {
    'num_leaves': [15, 31, 50],  
    'learning_rate': [0.01, 0.1, 0.5],  
    'n_estimators': [50, 100, 200]  
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],  
    'weights': ['uniform', 'distance'], 
    'metric': ['euclidean', 'manhattan']  
}

gbdt_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}

catboost_param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'depth': [4, 6, 8],
    'iterations': [50, 100, 200]
}

bart_param_grid = {
    'num_trees': [10, 50, 100],
    'alpha': [0.01, 0.1, 1.0],
    'beta': [0.01, 0.1, 1.0]
}

adaboost_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0]
}

gam_para_dict = {
    'lam': [0.1,0.3,0.5],  
    'n_splines': [10, 20, 30],  
}

logis_para_dict = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],  
}

# def get_data(sheet_id,x_index = -1,y_index = -1): # get data 
#     x,y = [],[]
#     sheet = file.sheet_by_index(sheet_id)
#     for i in range(1,sheet.nrows):  
#         user = sheet.row_values(i)
#         x.append(user[:x_index])
#         y.append(user[y_index])
#     return x,y
 
def get_params(selected_models_id):  # put all models' hyperparameters into a dict
    param_dict = []
    param_dict.append(gam_para_dict)
    param_dict.append(logis_para_dict)
    param_dict.append(adaboost_param_grid)  
    param_dict.append(catboost_param_grid)
    param_dict.append(gbdt_param_grid)
    param_dict.append(knn_param_grid)
    param_dict.append(lgb_param_grid)
    param_dict.append(rotation_forest_param_grid)
    param_dict.append(xgboost_param_grid)
    param_dict.append(rf_param_grid)
    param_dict.append(guassin_cls_param_grid)
    param_dict.append(extra_tree_param_grid)
    param_dict.append(decision_tree_param_dict)
    param_dict.append(svm_param_dict)
    
    param_dict_selected = []

    for id in selected_models_id:
        if id > 13: continue
        param_dict_selected.append(param_dict[id])

    return param_dict_selected

def get_models(selected_models_id):  # get all models
    models = []
    models.append(LogisticGAM())
    models.append(LogisticRegression(max_iter=1000))
    models.append(AdaBoostClassifier())

    models.append(CatBoostClassifier(iterations=1000,  
                                    learning_rate=0.1, 
                                    depth=6,  
                                    loss_function='Logloss', 
                                    eval_metric='Accuracy', 
                                    random_seed=42,  
                                    logging_level='Silent')  
    )
    models.append(GradientBoostingClassifier(n_estimators=100,  
                                            learning_rate=0.1,  
                                            max_depth=3,  
                                            random_state=42)  
    )
    models.append(KNeighborsClassifier(n_neighbors=5))
    models.append(lgb.LGBMClassifier())
    
    models.append(Pipeline([
        ('pca', PCA()),  
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))  
       
    ]))
    models.append(XGBClassifier())
    
    models.append(RandomForestClassifier(n_estimators=100, random_state=42))
    models.append(GaussianProcessClassifier(kernel=RBF(1.0), random_state=0))
    models.append(ExtraTreesClassifier())
    models.append(tree.DecisionTreeClassifier(random_state=40))
    models.append(svm.SVC(C=10, kernel='linear', probability=True))

    # selected_models_id = [model2id[name] for name in selected_models]
    models_selected = []

    for id in selected_models_id:
        if id > 13: continue
        models_selected.append(models[id])

    return models_selected

def find_params(X,y,selected_models_id):
    
    selected_models_names = [model_names[id] for id in selected_models_id]

    params_dict = {k:v for k,v in zip(selected_models_names,get_params(selected_models_id))}
    models = {k:v for k,v in zip(selected_models_names,get_models(selected_models_id))}
    best_params = {k:None for k in selected_models_names}

    for model_name in selected_models_names:
        
        if model_name in ['mlp','bayes']: continue
        model = models[model_name]
        param = params_dict[model_name]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.2,random_state=42)

        best_param = search(Xtrain,Ytrain,param,model)
        best_params[model_name] = best_param

    return best_params

def calculate_bca_ci(data, stat_func, alpha=0.05): # use bootstrap
    res = bootstrap(data, stat_func, confidence_level=1-alpha, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

def get_conf(y_prob,y_true,n_bootstrap):
    # ✅ 原始PRAUC
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    original_auc = auc(recall, precision)
    original_auc = min(original_auc,1)

    prauc = []
    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true)), replace=True)

        while (1 not in y_true[indices]) or (0 not in y_true[indices]):
            indices = resample(np.arange(len(y_true)), replace=True)
            
        y_true_bootstrap = y_true[indices]
        y_prob_bootstrap = y_prob[indices]

        precision, recall, thresholds = precision_recall_curve(y_true_bootstrap, y_prob_bootstrap)
        auc_score = auc(recall, precision)
        auc_score = min(auc_score,1)

        prauc.append(auc_score)
    
    low, high = calculate_bca_ci((prauc,),np.mean)
    return round(original_auc,3), round(low,3), round(high,3)  # ✅ 改为 original_auc

def train_model(X,y,X_val,y_val,selected_models):

    selected_models_id = [model2id[name] for name in selected_models]
    selected_models_id = sorted(selected_models_id)
    selected_models_names = [model_names[id] for id in selected_models_id]
    
    best_params = find_params(X,y,selected_models_id)

    models = []

    gam = LogisticGAM(**best_params['gam']) if best_params.get('gam') is not None else None
    logit = LogisticRegression(**best_params['logit']) if best_params.get('logit') is not None else None
    ada = AdaBoostClassifier(**best_params['ada']) if best_params.get('ada') is not None else None
    cat = CatBoostClassifier(**best_params['cat']) if best_params.get('cat') is not None else None
    gbdt = GradientBoostingClassifier(**best_params['gbdt']) if best_params.get('gbdt') is not None else None
    knn = KNeighborsClassifier(**best_params['knn']) if best_params.get('knn') is not None else None
    Lgb = lgb.LGBMClassifier(**best_params['lgb']) if best_params.get('lgb') is not None else None
    rotation = Pipeline([
        ('pca', PCA(n_components = best_params['rotation']['pca__n_components'])),  
        ('rf', RandomForestClassifier(n_estimators = best_params['rotation']['rf__n_estimators'], 
                                      max_depth = best_params['rotation']['rf__max_depth']))  
    ]) if best_params.get('rotation') is not None else None
    xgboost = XGBClassifier(**best_params['xgboost']) if best_params.get('xgboost') is not None else None
    rf = RandomForestClassifier(**best_params['rf']) if best_params.get('rf') is not None else None
    guassin_cls = GaussianProcessClassifier(**best_params['guassin_cls']) if best_params.get('guassin_cls') is not None else None
    extra_tree = ExtraTreesClassifier(**best_params['extra_tree']) if best_params.get('extra_tree') is not None else None
    dt = tree.DecisionTreeClassifier(**best_params['dt']) if best_params.get('dt') is not None else None
    Svm = svm.SVC(**best_params['svm']) if best_params.get('svm') is not None else None
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50),  
                            activation='relu',  
                            solver='adam',  
                            max_iter=1000, 
                            random_state=42) if "Multiply Layers Perception" in selected_models else None
    bayes = GaussianNB() if "Bayes" in selected_models else None

    models.append(gam)
    models.append(logit)
    models.append(ada)
    models.append(cat)
    models.append(gbdt)
    models.append(knn)
    models.append(Lgb)
    models.append(rotation)
    models.append(xgboost)
    models.append(rf)
    models.append(guassin_cls)
    models.append(extra_tree)
    models.append(dt)
    models.append(Svm)
    models.append(mlp)
    models.append(bayes)

    # results = {}
    best_thresholds = []
    model_data = {k:[] for k in sorted(selected_models,key=model2id.get)}

    value_name = ['acc','auc','tp','tn','fp','fn','sensitivity','specificity',
                  'pos_pred','neg_pred','f1','brier_score','prauc']
    
    # trained_models = []

    trained_models = {k:None for k in sorted(selected_models,key=model2id.get)}

    for name,model in zip(model_names,models):
        # auc = cross_val_score(model, X, y, scoring='roc_auc', cv=5).mean()
        # results[name] = auc
        if model is None: continue

        model.fit(X,y)

        name_trans = model_name_transfer[name]
        trained_models[name_trans] = model
        # trained_models.append(model)

        score = model.predict_proba(X_val)

        if name != 'gam':
            # print(score.shape)
            score = score[:,1]

        metrics, metrics_ci,best_threshold = bootstrap_metrics_bca(y_val, score)
            
        # print(best_threshold)
        best_thresholds.append(best_threshold)

        score = (score >= best_threshold)
        
        prauc,low,high = get_conf(score,y_val,2000)

        for key, value in metrics.items():
            
            v1 = metrics_ci[key][0]
            v2 = metrics_ci[key][1]

            if key not in ['tp','tn','fp','fn']:
                value = min(value,1)    
                v1 = min(metrics_ci[key][0],1)
                v2 = min(metrics_ci[key][1],1)

            if key in ['tp','tn','fp','fn']:
                v_range = f"{value:.3f}"
            else: v_range = f"{value:.3f}({v1:.3f},{v2:.3f})"

            model_data[name_trans].append(v_range.replace('nan','-'))

        model_data[name_trans].append(f"{prauc:.3f}({low:.3f},{high:.3f})")
        # sheet_data.append(model_data)   # 8 * 14

    # best_name = max(results, key=results.get)
    # best_model = models[best_name].fit(X, y)
    
    return pd.DataFrame.from_dict(model_data, orient='index', columns=value_name), trained_models


            # metrics, metrics_ci,best_threshold = bootstrap_metrics_bca(y, score,yoden_index=prob_standard[var_name[i]])
            
            # print(best_threshold)
            # best_thresholds.append(best_threshold)

            # score = (score >= prob_standard[var_name[i]])
            
            # prauc,low,high = get_conf(score,y,2000)

            # for key, value in metrics.items():
                
            #     v1 = metrics_ci[key][0]
            #     v2 = metrics_ci[key][1]

            #     if key not in ['tp','tn','fp','fn']:
            #         value = min(value,1)    
            #         v1 = min(metrics_ci[key][0],1)
            #         v2 = min(metrics_ci[key][1],1)

            #     if key in ['tp','tn','fp','fn']:
            #         v_range = f"{value:.3f}"
            #     else: v_range = f"{value:.3f}({v1:.3f},{v2:.3f})"

            #     model_data.append(v_range)

            # model_data.append(f"{prauc:.3f}({low:.3f},{high:.3f})")
            # sheet_data.append(model_data)   # 8 * 14
            
            # data_total.append(sheet_data)  # 1 * 8 * 14
# if __name__ == '__main__':

#     model_name = ['gam','logit','ada','cat','gbdt','knn','lgb', 
#                   'rotation','xgboost', 'rf','guassin_cls','extra_tree',
#                   'dt','svm']
    
#     params_dict = {k:v for k,v in zip(model_name,get_params())} 
#     params_dict
#     param_dict = get_params()
#     models = get_models()

#     model_name = [ 'gam','logit','ada','cat','gbdt','knn','lgb', 
#                   'rotation',
#                   'xgboost', 
#                   'rf','guassin_cls','extra_tree',
#                   'dt','svm',] #'mlp']
    
#     # model_name = ['mlp']
    
#     var_name = ['Total','intra_hem','Seizures','Reop','infection','Penu',
#                 'Fluid','Hydro']
    
#     num = 0
#     f = open('超参训练数据/总并发症单独超参.txt','a')

#     for model,param in zip(models,param_dict):
        
#         f.writelines('*'*6 + model_name[num] + ':' + '*'*6 +'\n')
#         print('*'*6 + model_name[num] + ':' + '*'*6)

#         for i in range(1):

#             X,y = get_data(i)
#             X = np.array(X)
#             y = np.array(y)

#             Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.2)

#             if model_name[num] == 'ngboost':
#                 Ytrain = np.array(Ytrain,dtype=np.int64)

#             print(str(i+1)+'th epoch data prepared!')
            
#             print('start grid search：')

#             f.writelines('\t' + var_name[i] + ":"+'\n')

#             print(var_name[i] + ":")
#             best_param = search(Xtrain,Ytrain,param,model)
#             f.writelines('\t\t' + str(best_param)+'\n')
#             print(best_param)
#             print('---------------------------------')
#             f.writelines('---------------------------------'+'\n')
#         num += 1
#         print('===========================================')
#         f.writelines('==========================================='+'\n')
    
#     f.close()   
