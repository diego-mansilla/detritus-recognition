import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)

data_augmentation = tf.keras.Sequential([
tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Custom simple CNN Models

def get_model_A(drop_value=0.0, data_aug=False, l2_reg=0.0):
    inputs = tf.keras.Input(shape=(160, 160, 3))
    
    if data_aug:
        dag = data_augmentation(inputs)
        processed_input =  tf.keras.applications.densenet.preprocess_input(dag)
    else:
        processed_input =  tf.keras.applications.densenet.preprocess_input(inputs)
    
    conv = keras.layers.Conv2D(filters=16, activation='relu', kernel_size=3)(processed_input)
    pool = keras.layers.MaxPooling2D(2,2)(conv)

    conv = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=3)(pool)
    pool = keras.layers.MaxPooling2D(2,2)(conv)

    conv = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=3)(pool)
    pool = keras.layers.MaxPooling2D(2,2)(conv)

    conv = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=3)(pool)
    pool = keras.layers.MaxPooling2D(2,2)(conv)

    conv = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=3)(pool)
    pool = keras.layers.MaxPooling2D(2,2)(conv)
    
    gal = keras.layers.Flatten()(pool)
    
    x = keras.layers.Dense(512)(gal)
    
    
    if drop_value > 0:
        x = keras.layers.Dropout(drop_value)(x)
    
    if l2_reg > 0:
        x = keras.layers.Dense(1, kernel_regularizer=regularizers.l2(l2_reg),
                bias_regularizer=regularizers.l2(l2_reg),
                activity_regularizer=regularizers.l2(l2_reg))(x)
    else:
        x = keras.layers.Dense(1)(x)
    return keras.Model(inputs, x)


def get_model_B(drop_value=0.0, data_aug=False, l2_reg=0.0):
    inputs = tf.keras.Input(shape=(160, 160, 3))
    
    if data_aug:
        dag = data_augmentation(inputs)
        processed_input =  tf.keras.applications.densenet.preprocess_input(dag)
    else:
        processed_input =  tf.keras.applications.densenet.preprocess_input(inputs)
    
    conv = keras.layers.Conv2D(filters=16, activation='relu', kernel_size=3)(processed_input)
    pool = keras.layers.MaxPooling2D(2,2)(conv)

    conv = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=3)(pool)
    pool = keras.layers.MaxPooling2D(2,2)(conv)

    conv = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=3)(pool)
    pool = keras.layers.MaxPooling2D(2,2)(conv)

    conv = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=3)(pool)
    pool = keras.layers.MaxPooling2D(2,2)(conv)

    conv = keras.layers.Conv2D(filters=64, activation='relu', kernel_size=3)(pool)
    pool = keras.layers.MaxPooling2D(2,2)(conv)
    
    x = keras.layers.GlobalAveragePooling2D()(pool)
    
    if drop_value > 0:
        x = keras.layers.Dropout(drop_value)(x)
    
    if l2_reg > 0:
        x = keras.layers.Dense(1, kernel_regularizer=regularizers.l2(l2_reg),
                bias_regularizer=regularizers.l2(l2_reg),
                activity_regularizer=regularizers.l2(l2_reg))(x)
    else:
        x = keras.layers.Dense(1)(x)
    return keras.Model(inputs, x)


def get_mobile_net(drop_value=0.0, data_aug=False, l2_reg=0.0):
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)

    inputs = tf.keras.Input(shape=(160, 160, 3))
    if data_aug:
        dag = data_augmentation(inputs)
        x =  tf.keras.applications.mobilenet_v2.preprocess_input(dag)
    else:
        x =  tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    if drop_value > 0:
        x = keras.layers.Dropout(drop_value)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return base_model, model


def get_dense_net(drop_value=0.0, data_aug=False, l2_reg=0.0):
    dense_base_model = tf.keras.applications.DenseNet201(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    dense_base_model.trainable = False

    dense_preprocess_input = tf.keras.applications.densenet.preprocess_input
    dense_global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense_prediction_layer = tf.keras.layers.Dense(1)

    dense_inputs = tf.keras.Input(shape=(160, 160, 3))
    if data_aug:
        dag = data_augmentation(dense_inputs)
        dense_x =  tf.keras.applications.densenet.preprocess_input(dag)
    else:
        dense_x =  tf.keras.applications.densenet.preprocess_input(dense_inputs)
    dense_x = dense_base_model(dense_x, training=False)
    dense_x = dense_global_average_layer(dense_x)
    if drop_value > 0:
        dense_x = keras.layers.Dropout(drop_value)(dense_x)
    dense_outputs = dense_prediction_layer(dense_x)
    dense_model = tf.keras.Model(dense_inputs, dense_outputs)
    return dense_base_model, dense_model
