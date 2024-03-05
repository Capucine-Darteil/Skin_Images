import numpy as np

import tensorflow as tf
from keras import Model, Sequential, layers, optimizers, callbacks


def initialize_dumb_model():
    model = Sequential()
    model.add(layers.Conv2D(16, (4,4), input_shape=(28, 28, 1), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam()
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

    return model


def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=30,
        validation_data=None, # overrides validation_split
        validation_split=0.3):
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = callbacks.EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64):
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True)

    return metrics
