import itertools
from matplotlib import rcParams
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE

from tqdm import tqdm

import random
random.seed(200)
np.random.seed(200)

# grid search for genetic algorithm
def grid_search(param_grid,X,y):
    
    y = y.astype(int)

    def eval_individual(individual,coef = 0.03):
        selected_features = [i for i in range(len(individual)) 
                            if individual[i] == 1]
        if len(selected_features) == 0:
            return 0.0,
        info_gain_sum = np.sum([info_gain[i] for i in selected_features])
        penalty = len(selected_features) * coef 
        score = info_gain_sum - penalty
        return score,

    info_gain = mutual_info_classif(X, y)
    feature_selection_frequency = np.zeros(len(info_gain))

    # define individual and population
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: np.random.randint(0, 2))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(info_gain))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    info_gain = mutual_info_classif(X, y)
    feature_selection_frequency = np.zeros(len(info_gain))
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", eval_individual)

    param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    best_score = float("-inf")
    best_params = None
    for params in param_combinations:
        print(f"Evaluating params: {params}")
        pop = toolbox.population(n=params['POP_SIZE'])
        
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(pop, toolbox, cxpb=params['CXPB'], mutpb=params['MUTPB'], ngen=params['NGEN'], 
                            stats=stats, halloffame=hof, verbose=False)

        X = np.array(X)
        y = np.array(y)
        
        best_individual = hof[0][:X.shape[-1]]
        selected_features = [i for i in range(len(best_individual)) 
                             if best_individual[i] == 1]
        if len(selected_features) == 0:
            continue
        
        for feature in selected_features:
            feature_selection_frequency[feature] += 1

        print(selected_features)

        score = eval_individual(best_individual)[0]

        if score > best_score:
            best_score = score
            best_params = params
            best_selected_features = selected_features

    return best_params, best_selected_features

def search(var_name,X,y):
    
    param_grid = {
        'POP_SIZE': [30, 50, 70],
        'NGEN': [30, 40, 50],
        'CXPB': [0.6, 0.7, 0.8],
        'MUTPB': [0.1, 0.2, 0.3],
    }
    
    best_params, best_selected_features = grid_search(param_grid,X,y)

    return [var_name[i] for i in best_selected_features]


def Lasso(X,y,var_names):
    lasso = LassoCV(cv=10, random_state=42).fit(X, y)
    # # 输出最佳 alpha
    # print("Best alpha:", lasso.alpha_)
    # 筛选特征
    coef = pd.Series(lasso.coef_, index=var_names[:-1])
    selected_features = coef[coef != 0].index.tolist()
    
    return selected_features

def recur(X,y):

    y = y.astype(int)

    if y.dtypes == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 初始化随机森林分类器
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 设置交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 训练随机森林并获取特征重要性
    rf.fit(X, y)
    feature_importances = rf.feature_importances_

    # 根据重要性排序特征
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_idx]

    # 保存每个特征子集的AUC
    auc_scores = []
    reverse = True
    # 逐步去除特征，按照重要性排序
    for i in tqdm(range(X.shape[1],0,-1)):

        selector = RFE(estimator=rf, n_features_to_select=i, step=1)
        X_selected = selector.fit_transform(X, y)


        # 使用交叉验证评估AUC
        auc_fold_scores = []
        for train_idx, test_idx in cv.split(X_selected, y):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 训练模型
            rf.fit(X_train, y_train)
            
            # 预测概率并计算AUC
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_fold_scores.append(auc)
        
        # 计算当前特征子集的平均AUC
        auc_scores.append(np.mean(auc_fold_scores))

    if reverse:
        auc_scores = auc_scores[::-1]

    # 找到最优特征数量的索引
    optimal_num_features = np.argmax(auc_scores) + 1

    # 输出最优特征数量和对应的特征
    optimal_features = sorted_features[:optimal_num_features]

    return optimal_features

def select_feature(methods,X,y,var_names):

    results = []

    for method in methods:
        if method == 'Genetic Algorithm':
            res = search(var_names,X,y)
            results.append(res)
        elif method == 'Lasso':
            res = Lasso(X,y,var_names)
            results.append(res)
        else:
            res = recur(X,y)
            results.append(res)
    
    return results[0] if len(results) == 1 else results

