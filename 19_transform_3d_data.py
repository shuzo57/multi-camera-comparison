import os
from itertools import combinations

import numpy as np
import pandas as pd
from IPython.display import display

from utils.dlt import *
from utils.skeleton_keypoints import *

keypoints_columns = [f"{keypoint}_{axis}" for keypoint in compare_keypoints_list for axis in ["x", "y", "z"]]

output_dir = "transformed_keypoints"
os.makedirs(output_dir, exist_ok=True)

for data_num in range(10):
    print(f"Processing hirasaki_{data_num}")
    true_path = f"fixed_trajectories_3/hirasaki_{data_num}_trajectories.csv"
    df_true = pd.read_csv(true_path, index_col=0)
    df_true = df_true[keypoints_columns]
    for n in range(5):
        for combo_indices in combinations(range(5), n+2):
            print(f"Processing combination {combo_indices}")
            pred_path = f"all_combinations/hirasaki_{data_num}_{''.join([str(i) for i in combo_indices])}.csv"
            df_pred = pd.read_csv(pred_path, index_col=0)
            df_pred = df_pred[keypoints_columns]

            index_max = min(df_true.index.max(), df_pred.index.max())
            index_min = max(df_true.index.min(), df_pred.index.min())

            df_pred = df_pred.loc[index_min:index_max]
            df_true = df_true.loc[index_min:index_max]

            df_X = pd.DataFrame(columns=keypoints_columns)
            df_Y = pd.DataFrame(columns=keypoints_columns)
            df_errors = pd.DataFrame(columns=keypoints_columns)

            for index in range(index_min, index_max+1):
                X = df_true.loc[index].values.reshape(-1, 3)
                Y = df_pred.loc[index].values.reshape(-1, 3)
                
                Y_transformed = procrustes_analysis_fixed_scale(X, Y)
                
                errors, x_errors, y_errors, z_errors = [], [], [], []
                
                for i in range(len(X)):
                    errors.append(np.linalg.norm(X[i] - Y_transformed[i]))
                    x_errors.append(np.abs(X[i][0] - Y_transformed[i][0]))
                    y_errors.append(np.abs(X[i][1] - Y_transformed[i][1]))
                    z_errors.append(np.abs(X[i][2] - Y_transformed[i][2]))
                errors = np.array(errors)
                x_errors = np.array(x_errors)
                y_errors = np.array(y_errors)
                z_errors = np.array(z_errors)

                df_X.loc[index] = X.reshape(-1)
                df_Y.loc[index] = Y_transformed.reshape(-1)
                df_errors.loc[index] = np.array([x_errors, y_errors, z_errors]).reshape(-1, 3).flatten()
            
            df_Y.to_csv(f"{output_dir}/hirasaki_{data_num}_{''.join([str(i) for i in combo_indices])}_transformed.csv")
            df_errors.to_csv(f"{output_dir}/hirasaki_{data_num}_{''.join([str(i) for i in combo_indices])}_errors.csv")
        df_X.to_csv(f"{output_dir}/hirasaki_{data_num}_true.csv")