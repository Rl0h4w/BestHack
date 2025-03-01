import os
import cv2
from collections import deque
import numpy as np


SIZE_OF_WINDOW = 30


class Dataset:
    def __init__(self, name, dir, size_of_window=30):
        self.name = name
        self.dir = dir
        self.size_of_window = size_of_window
        self.width = 320
        self.height = 240

    def __iter__(self):
        num_dirs = os.listdir(self.dir)
        for num_dir in num_dirs:
            subj_dir = os.path.join(self.dir, num_dir)
            subj_dir = os.path.join(subj_dir, os.listdir(subj_dir)[0])
            video_dir = os.path.join(subj_dir, "vid.avi")
            desc_dir = os.path.join(subj_dir, "ground_truth.txt")
            with open(desc_dir, "r") as file:
                trgts = deque([float(i) for i in file.readline().split()])
            cap = cv2.VideoCapture(video_dir)
            frame_window = deque(maxlen=self.size_of_window)
            trgts_window = deque(maxlen=self.size_of_window)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.width, self.height))
                frame_window.append(frame)
                trgts_window.append(trgts[frame_count])
                frame_count += 1
                if frame_count >= self.size_of_window:
                    yield np.array(frame_window), np.array(trgts_window)
