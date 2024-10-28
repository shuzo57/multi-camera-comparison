import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy

from utils.dlt import *
from utils.motion_visualize_tool import *
from utils.skeleton_keypoints import *

pred_fps = 60
true_fps = 250
keypoints_columns = [f"{keypoint}_{axis}" for keypoint in compare_keypoints_list for axis in ["x", "y", "z"]]
keypoints = ["RIGHT_ANKLE", "LEFT_ANKLE", "RIGHT_WRIST", "LEFT_WRIST"]
df_lag = pd.DataFrame(columns=[f"{col}_{keypoint}" for keypoint in keypoints for col in ["lag", "lag_time"]])

output_dir = "fig"
os.makedirs(output_dir, exist_ok=True)

for data_num in range(10):
    pred_path = f"all_combinations/hirasaki_{data_num}_0124.csv"
    true_path = f"fixed_trajectories_2/hirasaki_{data_num}_trajectories.csv"

    df_pred = pd.read_csv(pred_path, index_col=0)
    df_pred = df_pred[keypoints_columns]
    df_pred["time"] = np.arange(len(df_pred)) / pred_fps

    df_true = pd.read_csv(true_path, index_col=0)
    df_true = df_true[keypoints_columns]
    df_true["time"] = np.arange(len(df_true)) / true_fps

    pred_time = np.arange(0, len(df_pred)) / pred_fps
    true_time = np.arange(0, len(df_true)) / true_fps
    new_time = np.arange(0, pred_time[-1], 1 / true_fps)

    for keypoint in keypoints:
        y_true = df_true[f"{keypoint}_z"]
        y_pred = df_pred[f"{keypoint}_z"]

        interpolater = scipy.interpolate.interp1d(pred_time, y_pred, kind="linear")
        y_pred_interpolated = interpolater(new_time)

        y1 = y_true - y_true.mean()
        y2 = y_pred_interpolated - y_pred_interpolated.mean()

        corr = np.correlate(y1, y2, mode="full")
        lag = int(corr.argmax() - (len(y2) + 1))
        lag_time = lag / true_fps
        df_lag.loc[data_num, f"lag_{keypoint}"] = lag
        df_lag.loc[data_num, f"lag_time_{keypoint}"] = lag_time

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(df_true["time"] - lag_time, y_true, label="True")
        ax.plot(df_pred["time"], y_pred, label="Pred")
        plt.savefig(os.path.join(output_dir, f"corr_{data_num}_{keypoint}.png"))
        plt.close()
    
df_lag.to_csv("lag.csv", index=False)

