import os
import shutil
from itertools import combinations

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

from utils.skeleton_keypoints import *


def remove_outliers_iqr(data, factor=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[(data >= (Q1 - factor * IQR)) & (data <= (Q3 + factor * IQR))]

output_dir = "gif2"
os.makedirs(output_dir, exist_ok=True)

data_num = 0

true_path = f"transformed_keypoints/hirasaki_{data_num}_true.csv"
df_true = pd.read_csv(true_path, index_col=0)

for n in range(5):
    for combo_indices in combinations(range(5), n+2):
        comb = ''.join([str(i) for i in combo_indices])
        pred_path = f"transformed_keypoints/hirasaki_{data_num}_{comb}_transformed.csv"
        df_pred = pd.read_csv(pred_path, index_col=0)

        x_values = pd.concat([df_true.filter(like="_x"), df_pred.filter(like="_x")]).stack()
        y_values = pd.concat([df_true.filter(like="_y"), df_pred.filter(like="_y")]).stack()
        z_values = pd.concat([df_true.filter(like="_z"), df_pred.filter(like="_z")]).stack()

        max_x = remove_outliers_iqr(x_values).max()
        min_x = remove_outliers_iqr(x_values).min()
        max_y = remove_outliers_iqr(y_values).max()
        min_y = remove_outliers_iqr(y_values).min()
        max_z = remove_outliers_iqr(z_values).max()
        min_z = remove_outliers_iqr(z_values).min()

        os.makedirs('frames2', exist_ok=True)

        for idx in df_true.index[::2]:
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            ax.view_init(elev=10, azim=30)

            pose_result1 = df_true.loc[idx].values.reshape(-1, 3)
            pose_result2 = df_pred.loc[idx].values.reshape(-1, 3)
            
            ax.set_xlim3d(min_x, max_x)
            ax.set_ylim3d(min_y, max_y)
            ax.set_zlim3d(min_z, max_z)

            for key, value in compare_keypoints_connections.items():
                start_idx = value[0]
                end_idx = value[1]
                ax.plot([pose_result1[start_idx][0], pose_result1[end_idx][0]],
                        [pose_result1[start_idx][1], pose_result1[end_idx][1]],
                        [pose_result1[start_idx][2], pose_result1[end_idx][2]], linewidth=3, color='b', alpha=0.8)

            for i in range(len(pose_result1)):
                ax.scatter(pose_result1[i][0], pose_result1[i][1], pose_result1[i][2], color='r', s=8)

            for key, value in compare_keypoints_connections.items():
                start_idx = value[0]
                end_idx = value[1]
                ax.plot([pose_result2[start_idx][0], pose_result2[end_idx][0]],
                        [pose_result2[start_idx][1], pose_result2[end_idx][1]],
                        [pose_result2[start_idx][2], pose_result2[end_idx][2]], linewidth=3, color='g', alpha=0.8)

            for i in range(len(pose_result2)):
                ax.scatter(pose_result2[i][0], pose_result2[i][1], pose_result2[i][2], color='b', s=8)

            ax.plot([], [], color='g', label='Ground Truth')
            ax.plot([], [], color='b', label='Predicted')
            # ax.set_xlabel('X [mm]', labelpad=6)
            # ax.set_ylabel('Y [mm]', labelpad=6)
            # ax.set_zlabel('Z [mm]', labelpad=6)
            
            # ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            # ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            # ax.zaxis.set_major_locator(MaxNLocator(nbins=10))
            
            # 目盛りの数値だけ消す
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            
            ax.set_aspect('equal')
            # ax.legend(loc='upper right', fontsize=10)
            ax.set_title(f"{data_num}-{comb}", fontsize=12)
            plt.tight_layout()

            plt.savefig(f'frames2/frame_{idx}.png')
            plt.close(fig)

        with imageio.get_writer(f'{output_dir}/animation_{data_num}_{comb}.gif', mode='I', duration=0.1, loop=0) as writer:
            for idx in df_true.index[::2]:
                filename = f'frames/frame_{idx}.png'
                image = imageio.imread(filename)
                writer.append_data(image)

        shutil.rmtree('frames2')
