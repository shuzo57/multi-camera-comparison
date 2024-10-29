import os
from itertools import combinations
from typing import Union

import pandas as pd
import plotly.graph_objects as go

from .files import FileName
from .skeleton_keypoints import \
    compare_keypoints_connections as keypoints_connections
from .skeleton_keypoints import compare_keypoints_list as keypoints_list


def get_3d_motion_data(
    df: pd.DataFrame,
    frame: int,
) -> dict:
    target_dict = {}
    for part, (start, end) in keypoints_connections.items():
        start = keypoints_list[start]
        end = keypoints_list[end]
        for axis in ["x", "y", "z"]:
            key = f"{part}_{axis}"
            target_dict[key] = df[[f"{start}_{axis}", f"{end}_{axis}"]].loc[
                frame
            ]
    return target_dict


def plot_3d_motion_compare(
    threed_data: Union[pd.DataFrame, str],
    line_width: int = 10,
    marker_size: int = 3,
    graph_mode: str = "lines+markers",
    frame_step: int = 1,
    output_name: str = None,
    point_show: bool = False,
    club_show: bool = False,
    graph_show: bool = False,
) -> None:
    if output_name is None:
        output_name = FileName.threed_motion
    else:
        if not output_name.endswith(".html"):
            output_name += ".html"
    # Read the 3D motion data
    if isinstance(threed_data, pd.DataFrame):
        df = threed_data
        output_path = os.path.join(os.getcwd(), output_name)
    else:
        data_dir = os.path.dirname(threed_data)
        output_path = os.path.join(data_dir, output_name)

        df = pd.read_csv(threed_data)
        df.ffill(inplace=True)
        df = df.loc[(df.filter(like="_x").sum(axis=1) != 0)]

    if point_show:
        point_columns = [
            col.replace("_x", "")
            for col in df.columns
            if col.endswith("_x") and col.startswith("POINT")
        ]
        point_combinations = list(combinations(point_columns, 2))
        for i, (start, end) in enumerate(point_combinations):
            keypoints_connections[f"LINE{i+1}"] = [start, end]

    if club_show:
        keypoints_connections["SHAFT"] = ["GRIP", "HOSEL"]
        keypoints_connections["HEAD"] = ["HOSEL", "TOE"]

    # Create the 3D motion plot
    x_max = df.filter(like="_x").max().max()
    x_min = df.filter(like="_x").min().min()
    y_max = df.filter(like="_y").max().max()
    y_min = df.filter(like="_y").min().min()
    z_max = df.filter(like="_z").max().max()
    z_min = df.filter(like="_z").min().min()
    min_frame = df.index.min()
    max_frame = df.index.max()

    frames = []
    for frame in range(min_frame, max_frame + 1, frame_step):
        vec_data = get_3d_motion_data(df, frame)

        x_vec_label = list(vec_data.keys())[0::3]
        y_vec_label = list(vec_data.keys())[1::3]
        z_vec_label = list(vec_data.keys())[2::3]
        vec_name = [label.replace("_x", "") for label in x_vec_label]

        fig = go.Frame(
            data=[
                go.Scatter3d(
                    x=vec_data[x_label],
                    y=vec_data[y_label],
                    z=vec_data[z_label],
                    mode=graph_mode,
                    line=dict(width=line_width),
                    marker=dict(size=marker_size),
                    name=name,
                )
                for x_label, y_label, z_label, name in zip(
                    x_vec_label, y_vec_label, z_vec_label, vec_name
                )
            ],
            name=f"{frame}",
            layout=go.Layout(title=f"frame:{frame}"),
        )
        frames.append(fig)

    vec_data = get_3d_motion_data(df, min_frame)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vec_data[x_label],
                y=vec_data[y_label],
                z=vec_data[z_label],
                mode=graph_mode,
                line=dict(width=line_width),
                marker=dict(size=marker_size),
                name=name,
            )
            for x_label, y_label, z_label, name in zip(
                x_vec_label, y_vec_label, z_vec_label, vec_name
            )
        ],
        frames=frames,
    )

    steps = []
    for frame in range(min_frame, max_frame + 1, frame_step):
        step = dict(
            method="animate",
            args=[
                [f"{frame}"],
                dict(frame=dict(duration=1, redraw=True), mode="immediate"),
            ],
            label=f"{frame}",
        )
        steps.append(step)

    sliders = [
        dict(
            steps=steps,
            active=0,
            transition=dict(duration=0),
            currentvalue=dict(
                font=dict(size=20), prefix="", visible=True, xanchor="right"
            ),
        )
    ]

    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=x_max, y=y_max, z=z_max)),
            xaxis=dict(title="X", range=[x_min, x_max]),
            yaxis=dict(title="Y", range=[y_min, y_max]),
            zaxis=dict(title="Z", range=[z_min, z_max]),
            aspectmode="manual",
            aspectratio=dict(
                x=(x_max - x_min), y=(y_max - y_min), z=(z_max - z_min)
            ),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                xanchor="left",
                yanchor="top",
                x=0,
                y=1,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=sliders,
    )

    if graph_show:
        fig.update_layout(height=800, width=1200)
        fig.show()
    else:
        fig.write_html(output_path, auto_play=False)

def plot_3d_motion_double(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    file_name: str,
    line_width: int = 10,
    marker_size: int = 3,
    graph_mode: str = "lines+markers",
    frame_step: int = 1,
) -> None:
    x_max = max(
        df1.filter(like="_x").max().max(),
        df2.filter(like="_x").max().max(),
    )
    x_min = min(
        df1.filter(like="_x").min().min(),
        df2.filter(like="_x").min().min(),
    )
    y_max = max(
        df1.filter(like="_y").max().max(),
        df2.filter(like="_y").max().max(),
    )
    y_min = min(
        df1.filter(like="_y").min().min(),
        df2.filter(like="_y").min().min(),
    )
    z_max = max(
        df1.filter(like="_z").max().max(),
        df2.filter(like="_z").max().max(),
    )
    z_min = min(
        df1.filter(like="_z").min().min(),
        df2.filter(like="_z").min().min(),
    )
    min_frame = min(df1.index.min(), df2.index.min())
    max_frame = max(df1.index.max(), df2.index.max())

    frames = []
    for frame in range(min_frame, max_frame + 1, frame_step):
        vec_data1 = get_3d_motion_data(df1, frame)
        x_vec_label1 = list(vec_data1.keys())[0::3]
        y_vec_label1 = list(vec_data1.keys())[1::3]
        z_vec_label1 = list(vec_data1.keys())[2::3]
        vec_name1 = [label.replace("_x", "") for label in x_vec_label1]

        vec_data2 = get_3d_motion_data(df2, frame)
        x_vec_label2 = list(vec_data2.keys())[0::3]
        y_vec_label2 = list(vec_data2.keys())[1::3]
        z_vec_label2 = list(vec_data2.keys())[2::3]
        vec_name2 = [label.replace("_x", "") for label in x_vec_label2]

        fig = go.Frame(
            data=[
                go.Scatter3d(
                    x=vec_data1[x_label1],
                    y=vec_data1[y_label1],
                    z=vec_data1[z_label1],
                    mode=graph_mode,
                    line=dict(width=line_width),
                    marker=dict(size=marker_size, color='blue'),
                    name=name1,
                )
                for x_label1, y_label1, z_label1, name1 in zip(
                    x_vec_label1, y_vec_label1, z_vec_label1, vec_name1
                )
            ] + [
                go.Scatter3d(
                    x=vec_data2[x_label2],
                    y=vec_data2[y_label2],
                    z=vec_data2[z_label2],
                    mode=graph_mode,
                    line=dict(width=line_width),
                    marker=dict(size=marker_size, color='green'),
                    name=name2,
                )
                for x_label2, y_label2, z_label2, name2 in zip(
                    x_vec_label2, y_vec_label2, z_vec_label2, vec_name2
                )
            ],
            name=f"{frame}",
            layout=go.Layout(title=f"frame:{frame}"),
        )
        frames.append(fig)

    vec_data1 = get_3d_motion_data(df1, min_frame)
    vec_data2 = get_3d_motion_data(df2, min_frame)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vec_data1[x_label1],
                y=vec_data1[y_label1],
                z=vec_data1[z_label1],
                mode=graph_mode,
                line=dict(width=line_width),
                marker=dict(size=marker_size, color='blue'),
                name=name1,
            )
            for x_label1, y_label1, z_label1, name1 in zip(
                x_vec_label1, y_vec_label1, z_vec_label1, vec_name1
            )
        ] + [
            go.Scatter3d(
                x=vec_data2[x_label2],
                y=vec_data2[y_label2],
                z=vec_data2[z_label2],
                mode=graph_mode,
                line=dict(width=line_width),
                marker=dict(size=marker_size, color='green'),
                name=name2,
            )
            for x_label2, y_label2, z_label2, name2 in zip(
                x_vec_label2, y_vec_label2, z_vec_label2, vec_name2
            )
        ],
        frames=frames,
    )

    steps = []
    for frame in range(min_frame, max_frame + 1, frame_step):
        step = dict(
            method="animate",
            args=[
                [f"{frame}"],
                dict(frame=dict(duration=1, redraw=True), mode="immediate"),
            ],
            label=f"{frame}",
        )
        steps.append(step)

    sliders = [
        dict(
            steps=steps,
            active=0,
            transition=dict(duration=0),
            currentvalue=dict(
                font=dict(size=20), prefix="", visible=True, xanchor="right"
            ),
        )
    ]

    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=x_max, y=y_max, z=z_max)),
            xaxis=dict(title="X [m]", range=[x_min, x_max]),
            yaxis=dict(title="Y [m]", range=[y_min, y_max]),
            zaxis=dict(title="Z [m]", range=[z_min, z_max]),
            aspectmode="manual",
            aspectratio=dict(
                x=(x_max - x_min), y=(y_max - y_min), z=(z_max - z_min)
            ),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                xanchor="left",
                yanchor="top",
                x=0,
                y=1,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=sliders,
    )
    fig.write_html(f"{file_name}.html", auto_play=False)
    fig.show()
