import joblib
from matplotlib import rcParams
import shap
import numpy as np
import matplotlib.pyplot as plt

# draw combine figure
def draw_combine(X_combined,shap_values=None,sign = True):
    # import matplotlib.pyplot as plt
    # from matplotlib import rcParams

    if sign:
        shap_values = shap_values[1]  # 仅限二分类的第1类（正类）

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)

    print(shap_values.shape, len(X_combined), X_combined.columns)
    # 第一个 SHAP dot 图
    shap.summary_plot(shap_values, X_combined, feature_names=X_combined.columns, plot_type="dot", show=False, color_bar=True)
    ax1 = plt.gca()
    ax1.set_position([0.2, 0.2, 0.65, 0.65])

    # 添加第二个横轴用于 bar 图
    ax2 = ax1.twiny()
    shap.summary_plot(shap_values, X_combined, feature_names=X_combined.columns, plot_type="bar", show=False)
    ax2 = plt.gca()
    ax2.set_position([0.2, 0.2, 0.65, 0.65])
    ax2.axhline(y=13, color='gray', linestyle='-', linewidth=1)

    for bar in ax2.patches:
        bar.set_alpha(0.2)

    ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
    ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax1.set_ylabel('Features', fontsize=12)

    plt.tight_layout()
    rcParams['pdf.fonttype'] = 42

    return plt.gcf()  # 返回 Figure 对象

def draw(selected_test_name,models,X_combined):

    model = models[selected_test_name]
    
    if selected_test_name in ["Random Forest","Extra Tree","Gradient Boosting","Decision Tree"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_combined)
        shap_values = np.array(shap_values)

        fig = draw_combine(X_combined,shap_values=shap_values)
    else:
        explainer = shap.KernelExplainer(model.predict,X_combined,shap.kmeans(X_combined,10))
        shap_values = explainer.shap_values(X_combined)
        shap_values = np.array(shap_values)

        fig = draw_combine(X_combined,shap_values=shap_values,sign=False)

    return fig
