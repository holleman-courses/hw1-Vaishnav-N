#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import keras
from keras import Input, layers, Sequential

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


# ---------------------------------------------------------------------------
# Model builder functions
# ---------------------------------------------------------------------------

def build_model1():
  """Builds a four-layer fully-connected model: flatten, three Dense(128) with leaky ReLU, Dense(10) logits."""
  model = Sequential([
      layers.Flatten(input_shape=(32, 32, 3)),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(10),
  ])
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )
  return model


def build_model2():
  """Builds a CNN: two strided Conv2D blocks (32, 64 filters), four Conv2D(128) blocks with BatchNorm, Flatten, Dense(10)."""
  model = Sequential([
      layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Flatten(),
      layers.Dense(10),
  ])
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )
  return model


def build_model3():
  """Builds a CNN with the same topology as model2 using depthwise-separable convolutions and BatchNorm."""
  model = Sequential([
      layers.SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Flatten(),
      layers.Dense(10),
  ])
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )
  return model


def build_model50k():
  """Builds a CNN constrained to at most 50,000 parameters using Conv2D, SeparableConv2D, GlobalAveragePooling2D, and Dense layers."""
  model = Sequential([
      layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.GlobalAveragePooling2D(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10),
  ])
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )
  return model


if __name__ == '__main__':

  # Load CIFAR-10 and normalize pixel values to [0, 1].
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  train_images = train_images.astype(np.float32) / 255.0
  test_images = test_images.astype(np.float32) / 255.0

  # Reserve the last 10% of the training set for validation.
  n_val = int(0.1 * len(train_images))
  val_images = train_images[-n_val:]
  val_labels = train_labels[-n_val:]
  train_images = train_images[:-n_val]
  train_labels = train_labels[:-n_val]

  # Build, compile, and train the fully-connected model (model1) for 30 epochs.
  model1 = build_model1()
  model1.summary()
  model1.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  print('Model1 train:', model1.evaluate(train_images, train_labels))
  print('Model1 val:', model1.evaluate(val_images, val_labels))
  print('Model1 test:', model1.evaluate(test_images, test_labels))

  # Build, compile, and train the convolutional model (model2) for 30 epochs; save weights to disk.
  model2 = build_model2()
  model2.summary()
  model2.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  print('Model2 train/val/test:', model2.evaluate(train_images, train_labels), model2.evaluate(val_images, val_labels), model2.evaluate(test_images, test_labels))
  model2.save_weights('model2.h5')

  # Build, compile, and train the separable-convolution model (model3) for 30 epochs.
  model3 = build_model3()
  model3.summary()
  model3.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  print('Model3 train/val/test:', model3.evaluate(train_images, train_labels), model3.evaluate(val_images, val_labels), model3.evaluate(test_images, test_labels))

  # Load the custom test image, normalize to [0, 1], and run inference with model2.
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  test_img = np.array(keras.utils.load_img(
      './toyota.jpg',
      color_mode='rgb',
      target_size=(32, 32)))
  test_img = test_img.astype(np.float32) / 255.0
  pred = model2.predict(np.expand_dims(test_img, 0), verbose=0)
  predicted_class = class_names[np.argmax(pred[0])]
  print('Test image predicted class:', predicted_class)

  # Build, train, and save the parameter-constrained model (≤50k parameters) to best_model.h5.
  model50k = build_model50k()
  model50k.summary()
  assert model50k.count_params() <= 50000, 'Model must have at most 50k params'
  model50k.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  model50k.save('best_model.h5')
  loss, acc = model50k.evaluate(test_images, test_labels)
  print('Best model test accuracy:', acc)
