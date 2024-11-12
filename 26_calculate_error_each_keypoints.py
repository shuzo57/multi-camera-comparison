import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display

from utils.dlt import *
from utils.skeleton_keypoints import compare_keypoints_list

output_dir = "distance_error"
os.makedirs(output_dir, exist_ok=True)

for data_num in range(10):
    true_path = f"transformed_keypoints/hirasaki_{data_num}_true.csv"
    df_true = pd.read_csv(true_path, index_col=0)

    for n in range(5):
        for combo_indices in combinations(range(5), n+2):
            print(f"Calculating error for {data_num} with indices {combo_indices}")
            pred_path = f"transformed_keypoints/hirasaki_{data_num}_{''.join([str(i) for i in combo_indices])}_transformed.csv"
            df_pred = pd.read_csv(pred_path, index_col=0)

            df_diff = df_true - df_pred
            
            df_distance = pd.DataFrame(columns=compare_keypoints_list)
            for keyname in compare_keypoints_list:
                x = df_diff[f"{keyname}_x"]
                y = df_diff[f"{keyname}_y"]
                z = df_diff[f"{keyname}_z"]
                distance = np.sqrt(x**2 + y**2 + z**2)
                df_distance[keyname] = distance
            
            df_distance.to_csv(f"{output_dir}/hirasaki_{data_num}_{''.join([str(i) for i in combo_indices])}_distance_error.csv")