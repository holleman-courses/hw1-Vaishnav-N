#!/usr/bin/env python

# Reduce TensorFlow log noise (optional; remove if you want full logs)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=no INFO, 2=no WARNING, 3=errors only

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


##

def build_model1():
  """Fully-connected: Flatten + 3 Dense(128, leaky_relu) + Dense(10)."""
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
  """CNN: Conv2D(32,s2) -> BN -> Conv2D(64,s2) -> BN -> 4x (Conv2D(128) -> BN) -> Flatten -> Dense(10)."""
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
  """Same as model2 but all conv layers are SeparableConv2D (to match test spec)."""
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
  """Best model with no more than 50,000 parameters (for building/training in main)."""
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


# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  train_images = train_images.astype(np.float32) / 255.0
  test_images = test_images.astype(np.float32) / 255.0

  # Split training into train and validation (e.g. last 10% for validation)
  n_val = int(0.1 * len(train_images))
  val_images = train_images[-n_val:]
  val_labels = train_labels[-n_val:]
  train_images = train_images[:-n_val]
  train_labels = train_labels[:-n_val]
  ########################################

  ## Build and train model 1
  model1 = build_model1()
  model1.summary()
  model1.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  print('Model1 train:', model1.evaluate(train_images, train_labels))
  print('Model1 val:', model1.evaluate(val_images, val_labels))
  print('Model1 test:', model1.evaluate(test_images, test_labels))

  ## Build, compile, and train model 2
  model2 = build_model2()
  model2.summary()
  model2.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  print('Model2 train/val/test:', model2.evaluate(train_images, train_labels), model2.evaluate(val_images, val_labels), model2.evaluate(test_images, test_labels))

  ## Build, compile, and train model 3 (separable convolutions)
  model3 = build_model3()
  model3.summary()
  model3.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  print('Model3 train/val/test:', model3.evaluate(train_images, train_labels), model3.evaluate(val_images, val_labels), model3.evaluate(test_images, test_labels))

  ## Optional: load and classify a test image (save as test_image_<classname>.png/jpg)
  # class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  # test_img = np.array(keras.utils.load_img('./test_image_cat.png', grayscale=False, color_mode='rgb', target_size=(32,32)))
  # test_img = test_img.astype(np.float32) / 255.0
  # pred = model2.predict(np.expand_dims(test_img, 0))
  # print('Predicted class:', class_names[np.argmax(pred)])

  ## Best model (<=50k params), train and save
  model50k = build_model50k()
  model50k.summary()
  assert model50k.count_params() <= 50000, 'Model must have at most 50k params'
  model50k.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
  model50k.save('best_model.h5')
  loss, acc = model50k.evaluate(test_images, test_labels)
  print('Best model test accuracy:', acc)

  # plt.show()  # Commented out so script runs without intervention
