import json
import os
from itertools import combinations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from utils.dlt import *
from utils.files import FileName
from utils.motion_visualize_tool import *
from utils.skeleton_keypoints import *

output_dir = "all_combinations"
os.makedirs(output_dir, exist_ok=True)

with open("cube/subsets.json", "r") as f:
    subsets = json.load(f)
imgpoints = []
for i in ["0", "1", "2", "3", "4"]:
    imgpoints.append(subsets[f"{i}.mp4"])
imgpoints = np.array(imgpoints, dtype=np.float32)

with open("cube/3d_points.json", "r") as f:
    points = json.load(f)
cube_3d = np.array(points["object_point"], dtype=np.float32)
objpoints = np.array([cube_3d for _ in range(len(imgpoints))] , dtype=np.float32)

p1 = prepare_matrix(imgpoints[0], objpoints[0])
p2 = prepare_matrix(imgpoints[1], objpoints[1])
p3 = prepare_matrix(imgpoints[2], objpoints[2])
p4 = prepare_matrix(imgpoints[3], objpoints[3])
p5 = prepare_matrix(imgpoints[4], objpoints[4])

P_list = [p1, p2, p3, p4, p5]

for data_num in range(10):
    camera_name1 = "0"
    camera_name2 = "1"
    camera_name3 = "2"
    camera_name4 = "3"
    camera_name5 = "4"

    cam1_dir = f"data/hirasaki_{camera_name1}_{data_num}/"
    cam2_dir = f"data/hirasaki_{camera_name2}_{data_num}/"
    cam3_dir = f"data/hirasaki_{camera_name3}_{data_num}/"
    cam4_dir = f"data/hirasaki_{camera_name4}_{data_num}/"
    cam5_dir = f"data/hirasaki_{camera_name5}_{data_num}/"

    cam1_position = pd.read_csv(os.path.join(cam1_dir, FileName.position_data), index_col="frame")
    cam2_position = pd.read_csv(os.path.join(cam2_dir, FileName.position_data), index_col="frame")
    cam3_position = pd.read_csv(os.path.join(cam3_dir, FileName.position_data), index_col="frame")
    cam4_position = pd.read_csv(os.path.join(cam4_dir, FileName.position_data), index_col="frame")
    cam5_position = pd.read_csv(os.path.join(cam5_dir, FileName.position_data), index_col="frame")

    df_camera_sync = pd.read_csv("camera_sync.csv")

    toe_off_1 = df_camera_sync["toe_off_0"].iloc[data_num]
    toe_off_2 = df_camera_sync["toe_off_1"].iloc[data_num]
    toe_off_3 = df_camera_sync["toe_off_2"].iloc[data_num]
    toe_off_4 = df_camera_sync["toe_off_3"].iloc[data_num]
    toe_off_5 = df_camera_sync["toe_off_4"].iloc[data_num]

    delay_1 = toe_off_1 - toe_off_1
    delay_2 = toe_off_2 - toe_off_1
    delay_3 = toe_off_3 - toe_off_1
    delay_4 = toe_off_4 - toe_off_1
    delay_5 = toe_off_5 - toe_off_1

    cam1_position["ID"] = cam1_position.index - delay_1
    cam2_position["ID"] = cam2_position.index - delay_2
    cam3_position["ID"] = cam3_position.index - delay_3
    cam4_position["ID"] = cam4_position.index - delay_4
    cam5_position["ID"] = cam5_position.index - delay_5

    cam1_position = cam1_position.loc[cam1_position["ID"] > 0]
    cam2_position = cam2_position.loc[cam2_position["ID"] > 0]
    cam3_position = cam3_position.loc[cam3_position["ID"] > 0]
    cam4_position = cam4_position.loc[cam4_position["ID"] > 0]
    cam5_position = cam5_position.loc[cam5_position["ID"] > 0]

    cam1_position.set_index("ID", inplace=True, drop=True)
    cam2_position.set_index("ID", inplace=True, drop=True)
    cam3_position.set_index("ID", inplace=True, drop=True)
    cam4_position.set_index("ID", inplace=True, drop=True)
    cam5_position.set_index("ID", inplace=True, drop=True)

    min_frame = max(cam1_position.index.min(), cam2_position.index.min(), cam3_position.index.min(), cam4_position.index.min(), cam5_position.index.min())
    max_frame = min(cam1_position.index.max(), cam2_position.index.max(), cam3_position.index.max(), cam4_position.index.max(), cam5_position.index.max())

    cam_positions = [cam1_position, cam2_position, cam3_position, cam4_position, cam5_position]

    for n in range(5):
        for combo_indices in combinations(range(5), n+2):
            columns = [f"{kpt}_{xyz}" for kpt in exp_keypoints_list for xyz in ["x", "y", "z"]]
            position_df = pd.DataFrame(columns=columns)
            position_df.index.name = 'frame'
            
            cam = len(combo_indices)
            P_combo = np.array([P_list[i] for i in combo_indices])
            
            for frame in range(int(min_frame), int(max_frame)+1):
                poses = np.array([cam_positions[i].loc[frame].values.reshape(-1, 2) for i in combo_indices], dtype=np.float32)
                pose_result = pose_recon_2c(cam, P_combo, poses)
                position_df.loc[frame] = pose_result.ravel()
            
            position_df.to_csv(f"{output_dir}/hirasaki_{data_num}_{''.join([str(i) for i in combo_indices])}.csv")
            
            if cam == 5:
                plot_3d_motion_exp(position_df/1000, output_name=f"html/hirasaki_{data_num}.html")