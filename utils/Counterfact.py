import random
import dice_ml
from dice_ml import Dice
import joblib
from matplotlib import rcParams
import shap
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

seed = 500
random.seed(seed)
np.random.seed(seed)

def get_res(selected_test_name,models,X_combined,columns,continue_feature,
            features_to_vary,target_name,total_CFs=3):

        model = models[selected_test_name]

        data = dice_ml.Data(dataframe=X_combined,continuous_features = continue_feature,
                                outcome_name=target_name)

        model = dice_ml.Model(model=model,backend='sklearn')
        dice = Dice(data,model)
        X_0 = X_combined.loc[X_combined[target_name] == 1]
        id = X_0.index[random.randint(0,len(X_0.index))]

        # query_instance = X_combined.loc[[id]].to_dict()  # get random sample

        # fixed_feature_name = list(set(X_combined.columns) - set(features_to_vary))

        # columns = list(set(X_combined.columns) - {target_name})
        #     counterfactuals = dice.generate_counterfactuals(
        #                                 X_combined.loc[[id],X_combined.columns[:-1]], total_CFs=total_CFs, 
        #                                 features_to_vary=features_to_vary,)
        
        counterfactuals = dice.generate_counterfactuals(
                                        X_combined.loc[[id],columns], total_CFs=total_CFs, 
                                        features_to_vary=features_to_vary,)

        counterfactuals_df = counterfactuals.cf_examples_list[0].final_cfs_df
        counterfactuals_df.reset_index(drop=True, inplace=True)

        return counterfactuals_df

        
