import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display

from utils.dlt import *
from utils.skeleton_keypoints import angle_keypoints_dict


def get_mpjpe_error(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> list:
    mpjpe_errors = []
    for frame in df_true.index:
        true_points = df_true.loc[frame].values.reshape(-1, 3)
        pred_points = df_pred.loc[frame].values.reshape(-1, 3)

        mpjpe = MPJPE(true_points, pred_points)
        mpjpe_errors.append(mpjpe)
    return mpjpe_errors

def get_angle_error(df_true: pd.DataFrame, df_pred: pd.DataFrame, angle_keypoints_dict: dict) -> list:
    angle_errors_dict = {}
    for key, value in angle_keypoints_dict.items():
        a_column, b_column, c_column = value
        for frame in df_true.index:
            a = df_true.filter(like=a_column).loc[frame].values
            b = df_true.filter(like=b_column).loc[frame].values
            c = df_true.filter(like=c_column).loc[frame].values
            true_angle = calc_angle(a, b, c)
            
            a = df_pred.filter(like=a_column).loc[frame].values
            b = df_pred.filter(like=b_column).loc[frame].values
            c = df_pred.filter(like=c_column).loc[frame].values
            pred_angle = calc_angle(a, b, c)

            angle_error = np.abs(true_angle - pred_angle)
            if key not in angle_errors_dict:
                angle_errors_dict[key] = []
            angle_errors_dict[key].append(angle_error)
    return angle_errors_dict

output_dir = "error"
os.makedirs(output_dir, exist_ok=True)

for data_num in range(10):
    true_path = f"transformed_keypoints/hirasaki_{data_num}_true.csv"
    df_true = pd.read_csv(true_path, index_col=0)

    error_dict = {}

    for n in range(5):
        for combo_indices in combinations(range(5), n+2):
            print(f"Calculating error for {data_num} with indices {combo_indices}")
            pred_path = f"transformed_keypoints/hirasaki_{data_num}_{''.join([str(i) for i in combo_indices])}_transformed.csv"
            df_pred = pd.read_csv(pred_path, index_col=0)

            mpjpe_errors = get_mpjpe_error(df_true, df_pred)
            error_dict[f"{''.join([str(i) for i in combo_indices])}_mpjpe"] = mpjpe_errors
            
            angle_errors_dict = get_angle_error(df_true, df_pred, angle_keypoints_dict)
            for key, value in angle_errors_dict.items():
                error_dict[f"{''.join([str(i) for i in combo_indices])}_{key}_angle"] = value

    df_errors = pd.DataFrame(error_dict, index=df_true.index)
    df_errors.to_csv(os.path.join(output_dir, f"hirasaki_{data_num}_errors.csv"))