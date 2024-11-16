import json
import os
import time
from itertools import product

import numpy as np
import pandas as pd

from utils.dlt import *
from utils.files import FileName
from utils.motion_visualize_tool import *
from utils.skeleton_keypoints import *

# SEEDの固定
np.random.seed(42)

def get_mpjpe_error(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> list:
    mpjpe_errors = []
    for frame in df_true.index:
        true_points = df_true.loc[frame].values.reshape(-1, 3)
        pred_points = df_pred.loc[frame].values.reshape(-1, 3)
        mpjpe = MPJPE(true_points, pred_points)
        mpjpe_errors.append(mpjpe)
    return mpjpe_errors

def randomize_coordinates(imgpoints, range_limit, num_samples=10):
    random_samples = []
    for _ in range(num_samples):
        new_points = imgpoints.copy()
        for i in range(len(new_points)):
            offset_x = np.random.randint(-range_limit, range_limit + 1)
            offset_y = np.random.randint(-range_limit, range_limit + 1)
            new_points[i][0] += offset_x
            new_points[i][1] += offset_y
        random_samples.append(new_points)
    return random_samples

threshold_mean_mpjpe = 40
range_limit = 10
count = 0
best_imgpoints = None
best_mpjpe = float('inf')
finsihed = False

with open("cube/subsets.json", "r") as f:
    subsets = json.load(f)

imgpoints = []
for i in ["0", "3"]:
    imgpoints.append(subsets[f"{i}.mp4"])
imgpoints = np.array(imgpoints, dtype=np.float32)

while True:
    for data_num in range(10):
        count += 1
        start = time.time()
        candidate_points = randomize_coordinates(imgpoints[0], range_limit, num_samples=30)
        current_best_mpjpe = float('inf')
        current_best_imgpoints = None
        for candidate in candidate_points:
            imgpoints[0] = candidate
            with open("cube/3d_points.json", "r") as f:
                points = json.load(f)
            cube_3d = np.array(points["object_point"], dtype=np.float32)
            objpoints = np.array([cube_3d for _ in range(len(imgpoints))], dtype=np.float32)
            cam = 2
            p1 = prepare_matrix(imgpoints[0], objpoints[0])
            p2 = prepare_matrix(imgpoints[1], objpoints[1])
            P = np.array([p1, p2], dtype=np.float32)
            cube_array = pose_recon_2c(cam, P, imgpoints)
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
            columns = [f"{kpt}_{xyz}" for kpt in exp_keypoints_list for xyz in ["x", "y", "z"]]
            position_df = pd.DataFrame(columns=columns)
            position_df.index.name = 'frame'
            for frame in range(int(min_frame), int(max_frame) + 1):
                cam1_pose = cam1_position.loc[frame].values.reshape(-1, 2)
                cam4_pose = cam4_position.loc[frame].values.reshape(-1, 2)
                pose = np.array([cam1_pose, cam4_pose], dtype=np.float32)
                pose_result = pose_recon_2c(cam, P, pose)
                position_df.loc[frame] = pose_result.ravel()
            keypoints_columns = [f"{keypoint}_{axis}" for keypoint in compare_keypoints_list for axis in ["x", "y", "z"]]
            true_path = f"fixed_trajectories_3/hirasaki_{data_num}_trajectories.csv"
            df_true = pd.read_csv(true_path, index_col=0)
            df_true = df_true[keypoints_columns]
            df_pred = position_df[keypoints_columns]
            index_max = min(df_true.index.max(), df_pred.index.max())
            index_min = max(df_true.index.min(), df_pred.index.min())
            df_pred = df_pred.loc[index_min:index_max]
            df_true = df_true.loc[index_min:index_max]
            df_X = pd.DataFrame(columns=keypoints_columns)
            df_Y = pd.DataFrame(columns=keypoints_columns)
            for index in range(index_min, index_max + 1):
                X = df_true.loc[index].values.reshape(-1, 3)
                Y = df_pred.loc[index].values.reshape(-1, 3)
                Y_transformed = procrustes_analysis_fixed_scale(X, Y)
                df_X.loc[index] = X.reshape(-1)
                df_Y.loc[index] = Y_transformed.reshape(-1)
            mpjpe_errors = get_mpjpe_error(df_X, df_Y)
            mean_mpjpe = np.mean(mpjpe_errors)
            if mean_mpjpe < current_best_mpjpe:
                current_best_mpjpe = mean_mpjpe
                current_best_imgpoints = candidate
            if mean_mpjpe < best_mpjpe:
                best_mpjpe = mean_mpjpe
                best_imgpoints = candidate
            if mean_mpjpe < threshold_mean_mpjpe:
                end = time.time()
                print(f"Solution found! Iteration: {count} | Data: {data_num}")
                print(f"Time = {end - start:.2f} sec | Best MPJPE: {mean_mpjpe:.2f}")
                print("Updated imgpoints[0] = [")
                for point in imgpoints[0]:
                    print(f"    [{int(point[0])}, {int(point[1])}],")
                print("]")
                finsihed = True
                break
        if finsihed:
            break
        end = time.time()
        print(f"Iter {count} Data {data_num}: {end - start:.2f} (sec)")
        print(f"\tCurrent Best MPJPE: {current_best_mpjpe:.2f}")
        print(f"\tOverall Best MPJPE: {best_mpjpe:.2f}")
    if finsihed:
        break
    print("One epoch finished.")
    print(f"Best MPJPE: {best_mpjpe:.2f}")
    print("imgpoints[0]: = [")
    for point in best_imgpoints:
        print(f"    [{int(point[0])}, {int(point[1])}],")
    print("]")
    for data_num in range(10):
        imgpoints[0] = best_imgpoints
        with open("cube/3d_points.json", "r") as f:
            points = json.load(f)
        cube_3d = np.array(points["object_point"], dtype=np.float32)
        objpoints = np.array([cube_3d for _ in range(len(imgpoints))], dtype=np.float32)
        cam = 2
        p1 = prepare_matrix(imgpoints[0], objpoints[0])
        p2 = prepare_matrix(imgpoints[1], objpoints[1])
        P = np.array([p1, p2], dtype=np.float32)
        cube_array = pose_recon_2c(cam, P, imgpoints)
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
        columns = [f"{kpt}_{xyz}" for kpt in exp_keypoints_list for xyz in ["x", "y", "z"]]
        position_df = pd.DataFrame(columns=columns)
        position_df.index.name = 'frame'
        for frame in range(int(min_frame), int(max_frame) + 1):
            cam1_pose = cam1_position.loc[frame].values.reshape(-1, 2)
            cam4_pose = cam4_position.loc[frame].values.reshape(-1, 2)
            pose = np.array([cam1_pose, cam4_pose], dtype=np.float32)
            pose_result = pose_recon_2c(cam, P, pose)
            position_df.loc[frame] = pose_result.ravel()
        keypoints_columns = [f"{keypoint}_{axis}" for keypoint in compare_keypoints_list for axis in ["x", "y", "z"]]
        true_path = f"fixed_trajectories_3/hirasaki_{data_num}_trajectories.csv"
        df_true = pd.read_csv(true_path, index_col=0)
        df_true = df_true[keypoints_columns]
        df_pred = position_df[keypoints_columns]
        index_max = min(df_true.index.max(), df_pred.index.max())
        index_min = max(df_true.index.min(), df_pred.index.min())
        df_pred = df_pred.loc[index_min:index_max]
        df_true = df_true.loc[index_min:index_max]
        df_X = pd.DataFrame(columns=keypoints_columns)
        df_Y = pd.DataFrame(columns=keypoints_columns)
        for index in range(index_min, index_max + 1):
            X = df_true.loc[index].values.reshape(-1, 3)
            Y = df_pred.loc[index].values.reshape(-1, 3)
            Y_transformed = procrustes_analysis_fixed_scale(X, Y)
            df_X.loc[index] = X.reshape(-1)
            df_Y.loc[index] = Y_transformed.reshape(-1)
        mpjpe_errors = get_mpjpe_error(df_X, df_Y)
        mean_mpjpe = np.mean(mpjpe_errors)
        print(f"Data {data_num}: {mean_mpjpe:.2f}")
    
print("Finished!")
print("imgpoints[0] = [")
for point in best_imgpoints:
    print(f"    [{int(point[0])}, {int(point[1])}],")
print("]")
for data_num in range(10):
    imgpoints[0] = best_imgpoints
    with open("cube/3d_points.json", "r") as f:
        points = json.load(f)
    cube_3d = np.array(points["object_point"], dtype=np.float32)
    objpoints = np.array([cube_3d for _ in range(len(imgpoints))], dtype=np.float32)
    cam = 2
    p1 = prepare_matrix(imgpoints[0], objpoints[0])
    p2 = prepare_matrix(imgpoints[1], objpoints[1])
    P = np.array([p1, p2], dtype=np.float32)
    cube_array = pose_recon_2c(cam, P, imgpoints)
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
    columns = [f"{kpt}_{xyz}" for kpt in exp_keypoints_list for xyz in ["x", "y", "z"]]
    position_df = pd.DataFrame(columns=columns)
    position_df.index.name = 'frame'
    for frame in range(int(min_frame), int(max_frame) + 1):
        cam1_pose = cam1_position.loc[frame].values.reshape(-1, 2)
        cam4_pose = cam4_position.loc[frame].values.reshape(-1, 2)
        pose = np.array([cam1_pose, cam4_pose], dtype=np.float32)
        pose_result = pose_recon_2c(cam, P, pose)
        position_df.loc[frame] = pose_result.ravel()
    keypoints_columns = [f"{keypoint}_{axis}" for keypoint in compare_keypoints_list for axis in ["x", "y", "z"]]
    true_path = f"fixed_trajectories_3/hirasaki_{data_num}_trajectories.csv"
    df_true = pd.read_csv(true_path, index_col=0)
    df_true = df_true[keypoints_columns]
    df_pred = position_df[keypoints_columns]
    index_max = min(df_true.index.max(), df_pred.index.max())
    index_min = max(df_true.index.min(), df_pred.index.min())
    df_pred = df_pred.loc[index_min:index_max]
    df_true = df_true.loc[index_min:index_max]
    df_X = pd.DataFrame(columns=keypoints_columns)
    df_Y = pd.DataFrame(columns=keypoints_columns)
    for index in range(index_min, index_max + 1):
        X = df_true.loc[index].values.reshape(-1, 3)
        Y = df_pred.loc[index].values.reshape(-1, 3)
        Y_transformed = procrustes_analysis_fixed_scale(X, Y)
        df_X.loc[index] = X.reshape(-1)
        df_Y.loc[index] = Y_transformed.reshape(-1)
    mpjpe_errors = get_mpjpe_error(df_X, df_Y)
    mean_mpjpe = np.mean(mpjpe_errors)
    print(f"Data {data_num}: {mean_mpjpe:.2f}")