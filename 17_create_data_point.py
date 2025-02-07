import os
import shutil

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
df_lag = pd.read_csv("lag.csv")

fig_dir = "fig"
if os.path.exists(fig_dir):
    shutil.rmtree(fig_dir)
os.makedirs(fig_dir, exist_ok=True)

output_dir = "fixed_trajectories_3"
os.makedirs(output_dir, exist_ok=True)

for data_num in range(10):
    pred_path = f"all_combinations/hirasaki_{data_num}_01234.csv"
    true_path = f"fixed_trajectories_2/hirasaki_{data_num}_trajectories.csv"

    df_pred = pd.read_csv(pred_path, index_col=0)
    df_pred = df_pred[keypoints_columns]
    df_pred["time"] = np.arange(len(df_pred)) / pred_fps

    df_true = pd.read_csv(true_path, index_col=0)
    df_true = df_true[keypoints_columns]
    df_true["time"] = np.arange(len(df_true)) / true_fps

    lag = df_lag["lag_RIGHT_ANKLE"].values[data_num]
    lag_time = df_lag["lag_time_RIGHT_ANKLE"].values[data_num]

    pred_time = df_pred.time.values
    true_time = df_true.time.values - lag_time

    lower = np.where(pred_time >= pred_time[np.where(pred_time > true_time[0])].min())[0]
    upper = np.where(pred_time <= pred_time[np.where(pred_time < true_time[-1])].max())[0]
    common_index = np.intersect1d(lower, upper)
    new_time = pred_time[common_index]

    df_true2 = pd.DataFrame()
    for keypoint in keypoints_columns:
        y_true = df_true[keypoint]
        interpolater = scipy.interpolate.interp1d(true_time, y_true, kind="linear")
        y_true_interpolated = interpolater(new_time)
        df_true2[keypoint] = y_true_interpolated
    df_true2["time"] = new_time
    df_true2["frame"] = (df_true2["time"] * pred_fps + 0.5).astype(int) + 1
    df_true2.set_index("frame", inplace=True)
    
    df_true2.to_csv(f"{output_dir}/hirasaki_{data_num}_trajectories.csv")
    
    for keypoint in keypoints:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df_true2.index, df_true2[f"{keypoint}_z"], marker="o", label="true")
        ax.plot(df_pred.index, df_pred[f"{keypoint}_z"], marker="o", label="pred")
        ax.set_xlabel("frame")
        ax.set_ylabel("z [mm]")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/hirasaki_{data_num}_trajectories_{keypoint}.png")
        plt.close()
