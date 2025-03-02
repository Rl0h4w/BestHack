import os
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    BatchNormalization,
    Reshape,
    TimeDistributed,
    Dense,
    Dropout,
    LSTM,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from ultralytics import YOLO

train_data_dir = "/kaggle/input/sxuprl/train/train"


class VideoDataset:
    def __init__(self, data_dir, window_size=3):
        self.data_dir = data_dir
        self.window_size = window_size
        self.width, self.height = 160, 120
        self.face_model = YOLO(
            "/kaggle/input/face-detection-using-yolov8/yolov8n-face.pt"
        )

    def __iter__(self):
        for subject in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject)
            subject_path = os.path.join(subject_path, os.listdir(subject_path)[0])

            video_path = os.path.join(subject_path, "vid.avi")
            targets_path = os.path.join(subject_path, "ground_truth.txt")

            with open(targets_path, "r") as f:
                targets = np.array(list(map(float, f.readline().split())))
                targets = deque(targets)

            cap = cv2.VideoCapture(video_path)
            frame_buffer = deque(maxlen=self.window_size)
            target_buffer = deque(maxlen=self.window_size)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                orig_h, orig_w = frame.shape[:2]
                resized_for_yolo = cv2.resize(frame, (640, 480))
                results = self.face_model(resized_for_yolo, verbose=False)
                boxes = results[0].boxes.xyxy

                cropped_face = frame
                if boxes.shape[0] > 0:
                    boxes = boxes.cpu().numpy()
                    x1, y1, x2, y2 = boxes[0]

                    x_scale = orig_w / 640
                    y_scale = orig_h / 480
                    x1_orig = int(x1 * x_scale)
                    y1_orig = int(y1 * y_scale)
                    x2_orig = int(x2 * x_scale)
                    y2_orig = int(y2 * y_scale)

                    x1_orig = max(0, x1_orig)
                    y1_orig = max(0, y1_orig)
                    x2_orig = min(orig_w, x2_orig)
                    y2_orig = min(orig_h, y2_orig)

                    if x2_orig > x1_orig and y2_orig > y1_orig:
                        cropped_face = frame[y1_orig:y2_orig, x1_orig:x2_orig]

                resized_face = cv2.resize(
                    cropped_face,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                )
                resized_face = resized_face.astype(np.float32) / 255.0

                frame_buffer.append(resized_face)
                target_buffer.append(targets[frame_count])
                frame_count += 1

                if frame_count >= self.window_size:
                    yield np.array(frame_buffer), np.array(target_buffer)


def build_model(input_shape=(3, 120, 160, 3)):
    inputs = Input(shape=input_shape)

    x = Conv3D(32, (3, 3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling3D((1, 2, 2), padding="same")(x)

    x = Conv3D(64, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling3D((1, 2, 2), padding="same")(x)

    x = Conv3D(128, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling3D((1, 2, 2), padding="same")(x)

    x = Reshape((input_shape[0], -1))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, return_sequences=True)(x)

    x = TimeDistributed(Dense(256, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Dense(1))(x)
    outputs = Reshape((input_shape[0],))(x)

    model = Model(inputs, outputs)
    model.summary()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-5, decay_steps=10000, decay_rate=0.96
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


def create_dataset_pipeline(dataset):
    ds = tf.data.Dataset.from_generator(
        lambda: dataset,
        output_signature=(
            tf.TensorSpec(shape=(3, 120, 160, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
        ),
    )
    return ds.batch(400).prefetch(tf.data.AUTOTUNE)


train_ds = VideoDataset(train_data_dir)
test_ds = VideoDataset("/kaggle/input/sxuprl/test/test")

train_pipeline = create_dataset_pipeline(train_ds)
test_pipeline = create_dataset_pipeline(test_ds)

checkpoint = ModelCheckpoint(
    "best_model.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()

history = model.fit(
    train_pipeline, validation_data=test_pipeline, epochs=5, callbacks=[checkpoint]
)

best_model = tf.keras.models.load_model("best_model.keras")
