from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
import random

random.seed(100)
np.random.seed(100)
color = ['#f16c23', '#2b6a99', '#1b7c3d', 'lime', 'violet', 'yellow', 'blue', 'orange']

# 用于平滑ROC曲线
def smooth_roc_curve(fpr, tpr, num_points=100):
    fpr_smooth = np.linspace(0, 1, num_points)
    tpr_smooth = interp1d(fpr, tpr)(fpr_smooth)
    return fpr_smooth, tpr_smooth

# 用于绘制ROC曲线
def draw(y_score, y_true, var_name, model_name, color):

    data_name = ['Training Dataset', 'Test Dataset']

    # 创建图像对象
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(var_name + ' ROC', fontsize=14)

    lw = 2
    ax.plot([-0.02, 1], [-0.02, 1], color='#7f7f7f', lw=lw, linestyle='--', alpha=0.7)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    for n in range(len(y_score)):
        fpr, tpr, _ = roc_curve(y_true[n], y_score[n])
        fpr_smooth, tpr_smooth = smooth_roc_curve(fpr, tpr)
        fpr_smooth[0] = 0
        tpr_smooth[0] = 0
        roc_auc = roc_auc_score(y_true[n], y_score[n])
        ax.plot(fpr_smooth, tpr_smooth, color=color[n], lw=lw, alpha=0.7,
                label=f'{model_name} {data_name[n]} (AUC = {roc_auc:.2f})')

    ax.legend(loc="lower right", fontsize=12)
    rcParams['pdf.fonttype'] = 42
    fig.tight_layout()

    return fig

def draw_roc(selected_test_name,models,X,y,X_val,y_val,title):

    model = models[selected_test_name]
    y_prob_test = model.predict_proba(X_val)
    y_prob_train = model.predict_proba(X)

    if selected_test_name != "Logist GAM":
        y_prob_test = y_prob_test[:,1]
        y_prob_train = y_prob_train[:,1]

    y_true = [y,y_val]
    y_score = [y_prob_train,y_prob_test]
    fig = draw(y_score, y_true, title, selected_test_name, color)

    return fig

# if __name__ == '__main__':
#     # var_name = ['infection', 'Fluid', 'Penu', 'intra_hem', 'Hydro', 'Seizures', 'Total', 'Reop']
#     var_name = ['total']
#     # model_names = ['rf', 'extra_tree', 'rotation', 'rf', 'extra_tree', 'extra_tree', 'rf', 'rf']
#     model_names = ['rf']
#     data_path = ['递归消除特征/Derivation cohort (2).xlsx','递归消除特征/Derivation cohort (2).xlsx',
#                  '递归消除特征/External validation.xlsx'
#                  ]

#     color = ['#f16c23', '#2b6a99', '#1b7c3d', 'lime', 'violet', 'yellow', 'blue', 'orange']

#     for i, (vars, models) in enumerate(zip(var_name, model_names)):
#         y_true = []
#         y_score = []

#         print('-' * 6 + vars + ":" + '-' * 6)
       

#         for j,path in enumerate(data_path):
#             X, y = get_data(path=path, id=0)

#             if 'External' not in path and j == 0:   
#                 # test_rows = data_index[models][i].split(' ')
#                 # test_rows = get_rows(test_rows)
#                 # train_rows = list(set(list(range(len(X)))) - set(test_rows))

#                 y_t = []
#                 y_s = []
#                 k_folds = 5  # set K value 
#                 kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

#                 for train_index, test_index in kf.split(X):
#                     X_train, X_test = X[train_index], X[test_index]
#                     y_train, y_test = y[train_index], y[test_index]
#                     model = rf
#                     model.fit(X_train,y_train)
#                     score_ = model.predict_proba(X_test)[:,1]

#                     y_s.extend(list(score_))
#                     y_t.extend(list(y_test))

#                 y_true.append(np.array(y_t))
#                 y_score.append(np.array(y_s))

#             else:
#                 model = joblib.load(f'./model parameters/{models}_{vars}.pkl')
#                 y_prob = model.predict_proba(X)[:,1]
#                 y_true.append(y)
#                 y_score.append(y_prob)

#         print('-----------------------------------')
#         draw(y_score, y_true, vars, models, color)
