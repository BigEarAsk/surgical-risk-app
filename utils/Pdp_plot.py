import joblib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import fsolve
import pandas as pd
import shap
import xlrd


def draw(X,shap_values_df,feature):

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.scatter(X[feature], shap_values_df[feature], s=20, label='SHAP values', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-.', linewidth=1, label='SHAP = 0')

    if max(X[feature]) == 1 and min(X[feature]) == 0: 
        ax.set_xlim(-0.3, 1.3)
    # if feature in ['Pre.op.V.P', 'Pre.op.seizures', 'Pre.op.infection', 'Titanium']:
    #     ax.set_xlim(-0.3, 1.3)

    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel(f'SHAP value for\n{feature}', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend()
    plt.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42

    return fig

def draw_pdp(selected_test_name,models,X_combined,feature):
    model = models[selected_test_name]

    if selected_test_name in ["Random Forest","Extra Tree","Gradient Boosting","Decision Tree"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_combined)
        shap_values = np.array(shap_values)[1]

        # fig = draw_combine(X_combined,shap_values=shap_values)
    else:
        explainer = shap.KernelExplainer(model.predict,X_combined,shap.kmeans(X_combined,10))
        shap_values = explainer.shap_values(X_combined)
        shap_values = np.array(shap_values)

        # fig = draw_combine(X_combined,shap_values=shap_values,sign=False)
    values_df = pd.DataFrame(shap_values,columns=X_combined.columns)
    fig = draw(X_combined,values_df,feature)
    
    return fig
