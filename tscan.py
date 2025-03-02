import os
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from ultralytics import YOLO
from tqdm.keras import TqdmCallback
from tqdm import tqdm

WINDOW_SIZE = 150
HEIGHT, WIDTH = 64, 48
YOLO_BATCH_SIZE = 8
DATA_BATCH_SIZE = 2
PARALLEL_VIDEOS = 2


class VideoProcessor:
    def __init__(self):
        self.face_model = YOLO("yolov8n-face.pt")
        self.frame_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.target_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.height, self.width = HEIGHT, WIDTH

    def process_video(self, video_path, target_path):
        cap = cv2.VideoCapture(video_path.decode())
        with open(target_path.decode(), "r") as f:
            targets = np.array(list(map(float, f.readline().split())), dtype=np.float32)

        yolo_batch, original_frames = [], []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if yolo_batch:
                    self._process_batch(
                        yolo_batch, original_frames, targets, frame_count
                    )
                break

            original_frames.append(frame)
            yolo_batch.append(cv2.resize(frame, (640, 480)))

            if len(yolo_batch) == YOLO_BATCH_SIZE:
                self._process_batch(yolo_batch, original_frames, targets, frame_count)
                frame_count += len(yolo_batch)
                yolo_batch, original_frames = [], []

            while len(self.frame_buffer) >= WINDOW_SIZE:
                window_frames = list(self.frame_buffer)[:WINDOW_SIZE]
                window_targets = list(self.target_buffer)[:WINDOW_SIZE]

                for _ in range(WINDOW_SIZE // 2):
                    self.frame_buffer.popleft()
                    self.target_buffer.popleft()

                yield (
                    np.array(window_frames, dtype=np.float32),
                    np.array(window_targets, dtype=np.float32),
                )

        cap.release()

    def _process_batch(self, yolo_batch, original_frames, targets, start_idx):
        results = self.face_model(yolo_batch, verbose=False)
        for i, (frame, res) in enumerate(zip(original_frames, results)):
            if start_idx + i >= len(targets):
                break

            orig_h, orig_w = frame.shape[:2]
            boxes = res.boxes.xyxy.cpu().numpy()
            if boxes.shape[0] > 0:
                x1, y1, x2, y2 = boxes[0]
                x1 = int(x1 * orig_w / 640)
                y1 = int(y1 * orig_h / 480)
                x2 = int(x2 * orig_w / 640)
                y2 = int(y2 * orig_h / 480)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if x2 > x1 and y2 > y1:
                    frame = frame[y1:y2, x1:x2]

            resized = cv2.resize(frame, (self.width, self.height))
            resized = resized.astype(np.float32) / 255.0
            self.frame_buffer.append(resized)
            self.target_buffer.append(targets[start_idx + i])


def get_video_paths(data_dir):
    paths = []
    for subject in tqdm(os.listdir(data_dir), desc="Processing subjects"):
        subject_dir = os.path.join(
            data_dir, subject, os.listdir(os.path.join(data_dir, subject))[0]
        )
        paths.append(
            (
                os.path.join(subject_dir, "vid.avi").encode(),
                os.path.join(subject_dir, "ground_truth.txt").encode(),
            )
        )
    return paths


def create_dataset(data_dir):
    video_paths = get_video_paths(data_dir)
    ds = tf.data.Dataset.from_tensor_slices(video_paths)

    def load_video(paths):
        processor = VideoProcessor()
        return tf.data.Dataset.from_generator(
            processor.process_video,
            output_signature=(
                tf.TensorSpec(shape=(WINDOW_SIZE, HEIGHT, WIDTH, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(WINDOW_SIZE,), dtype=tf.float32),
            ),
            args=paths,
        )

    return (
        ds.interleave(
            load_video,
            cycle_length=PARALLEL_VIDEOS,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .batch(DATA_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


class TemporalShift(layers.Layer):
    def __init__(self, n_segment=8, fold_div=8, **kwargs):
        super().__init__(**kwargs)
        self.n_segment = n_segment
        self.fold_div = fold_div

    @tf.function
    def call(self, x):
        batch_size, time, h, w, c = tf.shape(x)

        pad = (self.n_segment - (time % self.n_segment)) % self.n_segment
        if pad > 0:
            x = tf.pad(x, [[0, 0], [0, pad], [0, 0], [0, 0], [0, 0]])
            time += pad

        x = tf.reshape(x, [-1, self.n_segment, time // self.n_segment, h, w, c])
        fold = c // self.fold_div

        out1, out2, out3 = tf.split(x, [fold, fold, c - 2 * fold], axis=-1)

        out1_shift = tf.concat([out1[:, 1:], tf.zeros_like(out1[:, :1])], axis=1)
        out2_shift = tf.concat([tf.zeros_like(out2[:, :1]), out2[:, :-1]], axis=1)

        x = tf.concat([out1_shift, out2_shift, out3], axis=-1)
        return tf.reshape(x, [batch_size, time, h, w, c])


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gap = layers.GlobalAveragePooling3D()
        self.dense1 = layers.Dense(units=1, activation="sigmoid")

    def call(self, x):
        att = self.gap(x)
        att = self.dense1(att)
        return x * tf.reshape(att, [-1, 1, 1, 1, 1])


def build_tscan_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, HEIGHT, WIDTH, 3))

    x = layers.Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D((1, 2, 2), strides=(1, 2, 2))(x)

    def res_block(x, filters):
        residual = (
            layers.Conv3D(filters, (1, 1, 1), strides=(1, 2, 2))(x)
            if x.shape[-1] != filters
            else x
        )
        x = TemporalShift()(x)
        x = layers.Conv3D(filters, (3, 3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(filters, (3, 3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = ChannelAttention()(x)
        return layers.ReLU()(x)

    x = res_block(x, 64)
    x = res_block(x, 128)
    x = res_block(x, 256)

    x = layers.TimeDistributed(layers.GlobalAvgPool2D())(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Conv1D(1, 3, padding="same")(x)
    outputs = layers.Reshape((WINDOW_SIZE,))(outputs)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4), loss="mae", metrics=["mae", "mse"]
    )
    return model


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_tscan_model()
    model.summary()

train_ds = create_dataset("/kaggle/input/sxuprl/train/train")
test_ds = create_dataset("/kaggle/input/sxuprl/test/test")

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    callbacks=[
        TqdmCallback(),
        callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        callbacks.EarlyStopping(patience=3),
    ],
)
