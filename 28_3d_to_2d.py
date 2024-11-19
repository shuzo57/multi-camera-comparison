import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from utils.skeleton_keypoints import compare_keypoints_list

output_dir = "annotations"
os.makedirs(output_dir, exist_ok=True)
keypoints_columns = [f"{keypoint}_{axis}" for keypoint in compare_keypoints_list for axis in ["x", "y"]]

df_sync = pd.read_csv("camera_sync.csv")

for camera_num in [0, 2, 3, 4]:
    for data_num in range(10):
        print(f"camera_num: {camera_num}, data_num: {data_num}")
        lag = df_sync.filter(like="toe").iloc[data_num, camera_num] - df_sync.filter(like="toe").iloc[data_num].min()

        df_3d = pd.read_csv(f"fixed_trajectories_3/hirasaki_{data_num}_trajectories.csv", index_col=0)
        del df_3d["time"]
        columns_basic = [s.replace("_x", "")for  s in df_3d.filter(regex="_x").columns]
        columns_xy = [f"{column}_{axis}" for column in columns_basic for axis in ["x", "y"]]

        with open(f"28/hirasaki_{camera_num}_extrinsic.json", "r") as f:
            extrinsic = json.load(f)
        mtx = np.array(extrinsic["mtx"])
        dist = np.array(extrinsic["dist"])
        rvec = np.array(extrinsic["rvec"])
        tvec = np.array(extrinsic["tvec"])

        points_3d_all = df_3d.values.reshape(-1, 3)
        points_2d_projected, _ = cv2.projectPoints(points_3d_all, rvec, tvec, mtx, None)
        points_2d_projected = points_2d_projected.reshape(-1, 2)
        points_2d_projected = points_2d_projected.reshape(-1, len(columns_basic)*2)

        df_2d_projected = pd.DataFrame(points_2d_projected, columns=columns_xy)
        df_2d_projected.index = df_3d.index
        df_2d_projected.to_csv(f"{output_dir}/hirasaki_{camera_num}_{data_num}.csv")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = None

        for idx in df_2d_projected.index:
            img_path = f"img/hirasaki_{camera_num}_{data_num}/hirasaki_{camera_num}_{data_num}_{idx+lag}.jpg"
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if video is None:
                video = cv2.VideoWriter(f"28/hirasaki_2d_{camera_num}_{data_num}_projected.mp4", fourcc, 30.0, (img.shape[1], img.shape[0]))
            for col in columns_basic:
                x, y = df_2d_projected.loc[idx, [col + "_x", col + "_y"]]
                cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        video.release()