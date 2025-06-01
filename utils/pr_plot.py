from scipy.stats import bootstrap
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import rcParams
from sklearn.utils import resample

color = [
        '#f16c23','#2b6a99',
        '#1b7c3d','lime','violet',
        'yellow','blue','orange',
        'pink','cyan','grey',
        'yellowgreen','purple','black'
    ]

def calculate_bca_ci(data, stat_func, alpha=0.05): # use bootstrap
    res = bootstrap(data, stat_func, confidence_level=1-alpha, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

def smooth_pr_curve(fpr, tpr, num_points=100):
    fpr_smooth = np.linspace(0, 1, num_points)
    tpr_smooth = interp1d(fpr, tpr)(fpr_smooth)
    return fpr_smooth, tpr_smooth

def get_conf(y_prob,y_true,n_bootstrap):

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
    return round(low,3),round(high,3)

def draw(y_score, y_true, var_name, model_name, color):
    data_name = ['Training Dataset', 'Test Dataset']

    # 创建图像对象
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(var_name + ' PR', fontsize=16)

    lw = 2
    ax.plot([0, 1], [0, 1], color='#7f7f7f', lw=lw, linestyle='--', alpha=0.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    for n in range(len(y_score)):
        precision, recall, _ = precision_recall_curve(y_true[n], y_score[n])
        auc_score = auc(recall, precision)
        auc_score = min(auc_score, 1.0)
        ax.plot(
            recall, precision,
            color=color[n],
            lw=lw,
            label=f'{model_name} {data_name[n]} (area = {auc_score:.2f})',
            alpha=0.8
        )

    ax.legend(loc="lower right", fontsize=12)
    # rcParams['pdf.fonttype'] = 42
    fig.tight_layout()
    return fig

def draw_pr(selected_test_name,models,X,y,X_val,y_val,title):

    model = models[selected_test_name]
    y_prob_test = model.predict_proba(X_val)
    y_prob_train = model.predict_proba(X)

    if selected_test_name != "Logist GAM":
        y_prob_test = y_prob_test[:,1]
        y_prob_train = y_prob_train[:,1]

    y_true = [y,y_val]
    y_score = [y_prob_train,y_prob_test]

    fig = draw(y_score,y_true,title,selected_test_name,color)

    return fig