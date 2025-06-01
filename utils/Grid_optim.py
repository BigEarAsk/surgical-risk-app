import random
import numpy as np
from sklearn import svm, tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import xlrd

# path = ''
# file = xlrd.open_workbook("one hot编码汇总数据.xlsx")
# Train_Rate = 0.8  # 训练数据占比
random.seed(100)
np.random.seed(100)

# def get_data(x_index = 16,y_index = -1):
#     x,y = [],[]
#     sheet = file.sheet_by_index(0)
#     for i in range(1,sheet.nrows):  # 数据行号
#         user = sheet.row_values(i)
#         x.append(user[:x_index])
#         y.append(user[y_index])
#     return x,y

def search(X=None,y=None,param_dict=None,model=None):
    # 加载示例数据集
    # iris = load_iris()
    # X, y = iris.data, iris.target

    # 定义要调优的参数空间
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }

    # 实例化模型
    # rf = RandomForestClassifier()

    # 使用 GridSearchCV 来搜索最佳参数
    grid_search = GridSearchCV(estimator=model, param_grid=param_dict, cv=5)
    grid_search.fit(X, y)

    # 输出最佳参数和对应的性能指标
    print("最佳参数：", grid_search.best_params_)
    print("最佳得分：", grid_search.best_score_)
    return grid_search.best_params_

def test(x,y,model):
    y_pred = model.predict(x)
    correct = np.sum(y_pred == y)
    acc = correct / len(y)
    print("预测正确的数目为：",correct)
    print("准确率为：%.2f(total=%d)"%(acc,len(y)))
    return y_pred,acc

# if __name__ == '__main__':
#     X,y = get_data()
#     X = np.array(X)
#     y = np.array(y)
#     Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=1-Train_Rate)
#     model = tree.DecisionTreeClassifier(random_state=40)
#     svm_model = svm.SVC(C=10, kernel='linear', probability=True)
#     param_dict = {
#         'criterion':["gini", "entropy", "log_loss"],
#         'max_depth':[None,5,10,15,20],
#         'min_samples_leaf':[1,2,3,4]
#     }
#     svm_param_dict = {
#         'C':[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.],
#         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#         'degree':[2,3,4,5],
#         'gamma':['scale','auto']
#     }
#     best_param = search(Xtrain,Ytrain,svm_param_dict,svm_model)

#     # model = tree.DecisionTreeClassifier(criterion=best_param['criterion'],
#     #                                     max_depth=best_param['max_depth'],
#     #                                     min_samples_leaf=best_param['min_samples_leaf'],
#     #                                     random_state=40)
#     svm_model = svm.SVC(C=best_param['C'], kernel=best_param['kernel'], 
#                         degree=best_param['degree'],gamma=best_param['gamma'],
#                         probability=True)
#     svm_model.fit(Xtrain,Ytrain)
#     test(Xtest,Ytest,svm_model)
    