import joblib
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import xlrd
import random
import statsmodels.api as sm

np.random.seed(100)
random.seed(100)

color = [
        'brown','#4CAF50','#2b6a99',
        'gold','lime','violet',
    ]

def smooth_calibration_curve(mean_values, frac_values, frac_std=None, smooth_frac_std=False): # smooth curve
    # use Loess curve
    lowess = sm.nonparametric.lowess
    smooth_frac = lowess(frac_values, mean_values, frac_std if frac_std is not None else 0.05)
    smooth_mean, smooth_frac = smooth_frac[:, 0], smooth_frac[:, 1]

    return smooth_mean, smooth_frac

def draw(mean_values,frac_values,var_name,model_name,color,train = 0): #  draw ROC curve

    line_name = ['Training Dataset',
                 'Test Dataset']

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(var_name + ' Calibration Curve', fontsize=14)
    lw = 2
    ax.plot([0, 1], [0, 1], color='#7f7f7f', lw=lw, linestyle='--', alpha=0.8, label='Ideal')

    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Mean predicted probability', fontsize=12)
    ax.set_ylabel('Fraction of positives', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    for i in range(len(mean_values)):
        smooth_mean, smooth_frac = smooth_calibration_curve(mean_values[i], frac_values[i], 0.85, 0.85)
        smooth_frac = np.clip(smooth_frac, 0, 1)
        ax.plot(smooth_mean, smooth_frac, color=color[i], lw=lw, marker='s',
                linestyle='-', label=line_name[i])

    ax.legend(loc="lower right", fontsize=12)

    fig.tight_layout()
    return fig


def draw_cali(selected_test_name,models,X,y,X_val,y_val,title):
    model = models[selected_test_name]
    y_prob_test = model.predict_proba(X_val)
    y_prob_train = model.predict_proba(X)

    if selected_test_name != "Logist GAM":
        y_prob_test = y_prob_test[:,1]
        y_prob_train = y_prob_train[:,1]

    mean = []
    frac = []

    true_rate, bin_means = calibration_curve(y, y_prob_train, n_bins=10)
    mean.append(bin_means)
    frac.append(true_rate)

    true_rate, bin_means = calibration_curve(y_val, y_prob_test, n_bins=10)
    mean.append(bin_means)
    frac.append(true_rate)

    fig = draw(mean,frac,title,selected_test_name,color)

    return fig