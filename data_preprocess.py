import os
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from ultralytics import YOLO

WINDOW_SIZE = 60
HEIGHT, WIDTH = 120, 160
YOLO_BATCH_SIZE = 16
DATA_BATCH_SIZE = 8
PARALLEL_VIDEOS = 4


class VideoProcessor:
    def __init__(self):
        self.face_model = YOLO(
            "/kaggle/input/face-detection-using-yolov8/yolov8n-face.pt"
        )
        self.frame_buffer = deque(maxlen=WINDOW_SIZE)
        self.target_buffer = deque(maxlen=WINDOW_SIZE)
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
                frame_count += YOLO_BATCH_SIZE
                yolo_batch, original_frames = [], []

            while len(self.frame_buffer) >= WINDOW_SIZE:
                yield (
                    np.array(list(self.frame_buffer)[-WINDOW_SIZE:], dtype=np.float32),
                    np.array(list(self.target_buffer)[-WINDOW_SIZE:], dtype=np.float32),
                )

        while len(self.frame_buffer) >= WINDOW_SIZE:
            yield (
                np.array(list(self.frame_buffer)[-WINDOW_SIZE:], dtype=np.float32),
                np.array(list(self.target_buffer)[-WINDOW_SIZE:], dtype=np.float32),
            )

        cap.release()

    def _process_batch(self, yolo_batch, original_frames, targets, start_idx):
        results = self.face_model(yolo_batch, verbose=False)
        for i, (frame, res) in enumerate(zip(original_frames, results)):
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
    for subject in os.listdir(data_dir):
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


def create_dataset(data_dir, num_parallel_calls=PARALLEL_VIDEOS):
    video_paths = get_video_paths(data_dir)
    ds = tf.data.Dataset.from_tensor_slices(video_paths)

    def load_video(paths):
        vp = paths[0]
        tp = paths[1]

        processor = VideoProcessor()
        return tf.data.Dataset.from_generator(
            processor.process_video,
            output_signature=(
                tf.TensorSpec(shape=(WINDOW_SIZE, HEIGHT, WIDTH, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(WINDOW_SIZE,), dtype=tf.float32),
            ),
            args=(vp, tp),
        )

    return (
        ds.interleave(
            load_video,
            cycle_length=num_parallel_calls,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .batch(DATA_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_3dcnn_lstm_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, HEIGHT, WIDTH, 3))

    x = layers.Conv3D(32, (3, 5, 5), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = layers.Conv3D(64, (3, 3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = layers.Conv3D(128, (3, 3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = layers.Reshape((WINDOW_SIZE, -1))(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)

    x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)
    outputs = layers.TimeDistributed(layers.Dense(1))(x)
    outputs = layers.Reshape((WINDOW_SIZE,))(outputs)

    model = tf.keras.Model(inputs, outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        3e-5, 10000, 0.9, staircase=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule, global_clipnorm=1.0),
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_3dcnn_lstm_model()
    model.summary()

train_ds = create_dataset("/kaggle/input/sxuprl/train/train")
test_ds = create_dataset("/kaggle/input/sxuprl/test/test")

callbacks_list = [
    callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor="val_root_mean_squared_error",
        mode="min",
    ),
    callbacks.TensorBoard(log_dir="./logs"),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,
    callbacks=callbacks_list,
)

best_model = tf.keras.models.load_model("best_model.keras")
test_loss = best_model.evaluate(test_ds)
print(f"Final Test Performance: {test_loss}")
