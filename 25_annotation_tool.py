import os
import tkinter as tk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from utils.skeleton_keypoints import compare_keypoints_list


class AnnotationTool:
    def __init__(self, video_path):
        self.root = tk.Tk()
        self.root.title("Annotation Tool")
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            self.root.destroy()
            return
        self.output_dir = "annotations"
        self.csv_path = os.path.join(self.output_dir, os.path.basename(self.video_path).replace(".mp4", ".csv"))
        self.frame_index = 1
        self.label_index = 0
        self.image_label = None
        self.click_points = {compare_keypoints_list[self.label_index]: {}}
        self.df = pd.DataFrame()
        self.create_widgets()
        self.update_frame()
        self.root.bind_all("d", self.move_left)
        self.root.bind_all("a", self.move_right)
        self.root.bind_all("w", self.next_label)
        self.root.bind_all("s", self.previous_label)
        self.root.bind_all("q", self.quit_app)
        self.image_label.bind("<Button-1>", self.record_click)
        self.image_label.bind("<Button-3>", self.record_click_out_of_view)
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path, index_col="frame")
        else:
            self.set_dataframe()

    def create_widgets(self):
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index - 1)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_label = compare_keypoints_list[self.label_index]
            cv2.putText(frame, f"Frame: {self.frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Label: {current_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if self.frame_index - 1 in self.click_points[current_label]:
                point = self.click_points[current_label][self.frame_index - 1]
                color = (0, 255, 0) if point[2] == 'green' else (255, 0, 0)
                cv2.circle(frame, (point[0], point[1]), 5, color, -1)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def move_left(self, event):
        self.frame_index = max(self.frame_index - 1, 0)
        self.update_frame()

    def move_right(self, event):
        self.frame_index = min(self.frame_index + 1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.update_frame()

    def next_label(self, event):
        self.label_index = (self.label_index + 1) % len(compare_keypoints_list)
        if compare_keypoints_list[self.label_index] not in self.click_points:
            self.click_points[compare_keypoints_list[self.label_index]] = {}
        self.frame_index = 1
        self.update_frame()

    def previous_label(self, event):
        self.label_index = (self.label_index - 1) % len(compare_keypoints_list)
        if compare_keypoints_list[self.label_index] not in self.click_points:
            self.click_points[compare_keypoints_list[self.label_index]] = {}
        self.frame_index = 1
        self.update_frame()

    def record_click(self, event):
        x, y = event.x, event.y
        current_label = compare_keypoints_list[self.label_index]
        self.click_points[current_label][self.frame_index] = [x, y, 'green']
        self.move_right(event)

    def record_click_out_of_view(self, event):
        x, y = event.x, event.y
        current_label = compare_keypoints_list[self.label_index]
        self.click_points[current_label][self.frame_index] = [x, y, 'red']
        self.move_right(event)
    
    def set_dataframe(self):
        self.df = pd.DataFrame(columns=[f"{keyname}_{suffix}" for keyname in compare_keypoints_list for suffix in ["x", "y", "visibility"]], index=np.arange(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1))
        self.df.index.name = "frame"
    
    def write_dataframe(self):
        for keyname, points in self.click_points.items():
            for frame, point in points.items():
                self.df.loc[frame, f"{keyname}_x"] = point[0]
                self.df.loc[frame, f"{keyname}_y"] = point[1]
                self.df.loc[frame, f"{keyname}_visibility"] = 1 if point[2] == 'green' else 2
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.df.to_csv(self.csv_path, index=True)

    def quit_app(self, event):
        self.write_dataframe()
        self.root.quit()
        self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    video_path = "walk/hirasaki_0_0.mp4"
    if video_path:
        app = AnnotationTool(video_path)
        app.run()
