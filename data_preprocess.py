import os
import cv2
from collections import deque
import numpy as np
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


class Dataset:
    def __init__(self, name, dir, size_of_window=60):
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
            target_window = deque(maxlen=self.size_of_window)
            frame_count = 0
            print(num_dir)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (self.width, self.height))
                frame_window.append(frame)
                target_window.append(trgts[frame_count])
                frame_count += 1

                if frame_count >= self.size_of_window:
                    yield np.array(frame_window) / 255, np.array(target_window)


def get_model(input_shape=(60, 240, 320, 3)):
    inputs = Input(shape=input_shape)

    # Conv3D Block 1
    x = Conv3D(32, (3, 3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling3D((1, 2, 2), padding="same")(x)

    # Conv3D Block 2
    x = Conv3D(64, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling3D((1, 2, 2), padding="same")(x)

    # Conv3D Block 3
    x = Conv3D(128, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling3D((1, 2, 2), padding="same")(x)

    # Temporal processing
    x = Reshape((input_shape[0], -1))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, return_sequences=True)(x)

    # Regression head
    x = TimeDistributed(Dense(256, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Dense(1))(x)
    outputs = Reshape((input_shape[0],))(x)

    model = Model(inputs, outputs)

    # Optimizer with lower learning rate and gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
with strategy.scope():
    model = get_model(input_shape=(60, 240, 320, 3))

train_ds = Dataset("train", "/kaggle/input/sxuprl/train/train")
test_ds = Dataset("test", "/kaggle/input/sxuprl/test/test")


def create_tf_dataset(dataset):
    return tf.data.Dataset.from_generator(
        lambda: dataset,
        output_signature=(
            tf.TensorSpec(shape=(60, 240, 320, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(60,), dtype=tf.float32),
        ),
    ).batch(4)


train_dataset = create_tf_dataset(train_ds)
test_dataset = create_tf_dataset(test_ds)
checkpoint = ModelCheckpoint(
    filepath="model.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=1,
)
model.fit(train_dataset, validation_data=test_dataset, callbacks=[checkpoint])
