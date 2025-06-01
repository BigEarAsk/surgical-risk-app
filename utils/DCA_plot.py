import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import io

data_name = ['Training Dataset','Test Dataset']
color = ['crimson','#4CAF50','blue']
none_color = ['black','#996633','#7f7f7f']
all_color = ['black','#996633','#7f7f7f']

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label): # calculate net benefit model
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label): # calculate net benefit all
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all,data_name,color,
             none_color,all_color):  # plot DCA figure

    for i in range(len(thresh_group)):

        # Plot
        ax.plot(thresh_group[i], net_benefit_model[i], color[i], label = data_name[i])
        ax.plot(thresh_group[i], net_benefit_all[i], color = all_color[i],label = data_name[i] + ' Treat all')
        ax.plot((0, 1), (0, 0), color = none_color[i], linestyle = ':', label = data_name[i] + ' Treat none')

        # fill to show the better part model than all and none
        y2 = np.maximum(net_benefit_all[i], 0)
        y1 = np.maximum(net_benefit_model[i], y2)
        ax.fill_between(thresh_group[i], y1, y2, color = color[i], alpha = 0.2)

        # Figure Configuration
        ax.set_xlim(0,1)
        ax.set_ylim(net_benefit_model[i].min() - 0.15, net_benefit_model[i].max() + 0.15)
        # adjustify the y axis limitation

    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'fontsize': 15}
        )
    
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'fontsize': 15}
        )
    
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax

def draw_DCA(selected_test_name,models,X,y,X_val,y_val,title):

    thresh_groups = []
    net_benefit_models = []
    net_benefit_alls = []
    
    model = models[selected_test_name]
    y_prob_test = model.predict_proba(X_val)
    y_prob_train = model.predict_proba(X)

    if selected_test_name != "Logist GAM":
        y_prob_test = y_prob_test[:,1]
        y_prob_train = y_prob_train[:,1]
    
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_prob_train, y)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y)
    
    thresh_groups.append(thresh_group)
    net_benefit_alls.append(net_benefit_all)
    net_benefit_models.append(net_benefit_model)

    thresh_group = np.arange(0,1,0.01)
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_prob_test, y_val)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_val)
    
    thresh_groups.append(thresh_group)
    net_benefit_alls.append(net_benefit_all)
    net_benefit_models.append(net_benefit_model)

    fig, ax = plt.subplots()
    plt.title(f'{title} DCA Curve')
    ax = plot_DCA(ax, thresh_groups, net_benefit_models, net_benefit_alls, data_name, color,
                    none_color, all_color)
    

    # rcParams['pdf.fonttype'] = 42   
    # fig.savefig('./DCA/'+vars + ' DCA.pdf')
    # plt.show()

    return fig

# if __name__ == '__main__':

#     var_name = ['infection','Fluid','Penu','intra_hem','Hydro','Seizures',
#                 'Total','Reop']
    
#     model_names = ['rf','extra_tree',
#                    'rotation','rf',
#                    'extra_tree',
#                    'extra_tree','rf','rf',]
    
#     # data_path = ['AUC/Derivation Cohort(After Genetic Algorithm).xlsx',
#     #              'AUC/External Validation Cohort(After Genetic Algorithm).xlsx'
#     #              ]
    
#     data_path = ['Model explanation/Derivation Cohort(After Genetic Algorithm).xlsx',
#                  'Model explanation/External Validation Cohort(After Genetic Algorithm).xlsx'
#                  ]
    
#     data_name = ['Internal Validations Cohort','Training Cohort','External Validations Cohort']
#     color = ['crimson','#4CAF50','blue']
#     none_color = ['black','#996633','#7f7f7f']
#     all_color = ['black','#996633','#7f7f7f']

#     for i,(vars,model_name) in enumerate(zip(var_name,model_names)):
#         if model_name == 'mlp': continue
        
#         thresh_groups = []
#         net_benefit_models = []
#         net_benefit_alls = []

#         for path in data_path:
#             X,y = get_data(path=path,id=i)
#             print('-'*6 + vars +":" + '-'*6)
#             print(model_name)
#             model = joblib.load('./model parameters/'+model_name+'_'+vars+'.pkl')

#             if 'External' not in path:    # split into test set
#                 test_rows = data_index[model_name][i].split(' ')
#                 test_rows = get_rows(test_rows)
#                 train_rows = list(set(list(range(len(X)))) - set(test_rows))
            
#                 y_prob_test = model.predict_proba(X[test_rows])
#                 y_prob_train = model.predict_proba(X[train_rows])

#                 if model_name != 'gam':
#                     y_prob_test = y_prob_test[:,1]
#                     y_prob_train = y_prob_train[:,1]

#                 thresh_group = np.arange(0,1,0.01)
#                 net_benefit_model = calculate_net_benefit_model(thresh_group, y_prob_test, y[test_rows])
#                 net_benefit_all = calculate_net_benefit_all(thresh_group, y[test_rows])
                
#                 thresh_groups.append(thresh_group)
#                 net_benefit_alls.append(net_benefit_all)
#                 net_benefit_models.append(net_benefit_model)
                
#                 # train set
#                 net_benefit_model = calculate_net_benefit_model(thresh_group, y_prob_train, y[train_rows])
#                 net_benefit_all = calculate_net_benefit_all(thresh_group, y[train_rows])
                
#                 thresh_groups.append(thresh_group)
#                 net_benefit_alls.append(net_benefit_all)
#                 net_benefit_models.append(net_benefit_model)

#             else:
#                 y_prob = model.predict_proba(X)
#                 if model_name != 'gam':
#                     y_prob = y_prob[:,1]
#                 thresh_group = np.arange(0,1,0.01)
#                 net_benefit_model = calculate_net_benefit_model(thresh_group, y_prob, y)
#                 net_benefit_all = calculate_net_benefit_all(thresh_group, y)

#                 thresh_groups.append(thresh_group)
#                 net_benefit_alls.append(net_benefit_all)
#                 net_benefit_models.append(net_benefit_model)

#         fig, ax = plt.subplots()
#         plt.title(vars + ' DCA Curve')
#         ax = plot_DCA(ax, thresh_groups, net_benefit_models, net_benefit_alls, data_name, color,
#                       none_color, all_color)
        

#         rcParams['pdf.fonttype'] = 42   
#         fig.savefig('./DCA/'+vars + ' DCA.pdf')
#         plt.show()

