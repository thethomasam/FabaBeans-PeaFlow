#!/usr/bin/env python3

import cv2
import numpy as np
import os
import pandas as pd
# import tensorflow as tf

from sklearn.model_selection import train_test_split


IMG_COLS = 64
IMG_ROWS = 64
NUM_CLASSES = 2


def load_data(use_bfloat16=False):
    """Loads the FabaBeans 64 dataset and creates train and eval dataset objects.

    Args:
        use_bfloat16: Boolean to determine if input should be cast to `bfloat16`.

    Returns:
        Train dataset, eval dataset and input shape.

    """

    cast_dtype = tf.bfloat16 if use_bfloat16 else tf.float32

    x = []
    y = []
    annotations = pd.read_csv("fababeans-64/annotations.csv")
    for index, row in annotations.iterrows():
        x.append(cv2.imread(os.path.join("fababeans-64", row["path"])))
        y.append(row["label"])
    x = np.asarray(x)
    y = np.asarray(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    if tf.keras.backend.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 3, IMG_ROWS, IMG_COLS)
        x_test = x_test.reshape(x_test.shape[0], 3, IMG_ROWS, IMG_COLS)
        input_shape = (3, IMG_ROWS, IMG_COLS)
    else:
        x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 3)
        x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 3)
        input_shape = (IMG_ROWS, IMG_COLS, 3)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    train_ds = train_ds.batch(64, drop_remainder=True)

    # eval dataset
    eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    eval_ds = eval_ds.repeat()
    eval_ds = eval_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    eval_ds = eval_ds.batch(64, drop_remainder=True)

    return train_ds, eval_ds, input_shape


def load_model(input_shape):
    return tf.keras.models.Sequential(
        # [
        #     tf.keras.layers.Conv2D(
        #         32, (3, 3), activation=tf.nn.relu, input_shape=input_shape
        #     ),
        #     tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #     tf.keras.layers.Dropout(0.25),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation=tf.nn.relu),
        #     tf.keras.layers.Dropout(0.5),
        #     tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax),
        # ]
        [
            tf.keras.layers.Conv2D(
                64, (3, 3), activation=tf.nn.relu, input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax),
        ]
    )


def main():
    train_ds, eval_ds, input_shape = load_data()

    model = load_model(input_shape)

    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(x=train_ds, epochs=5, steps_per_epoch=500)

    model.save("pnn-cnn.h5")


if __name__ == "__main__":
    main()
