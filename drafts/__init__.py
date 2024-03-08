from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from Skin_Project.ml_logic.model_cat import compile_model, train_model, evaluate_model, initialize_model
from Skin_Project.ml_logic.preprocess import labelize, sampler, drop_columns, categorize
from keras.utils import to_categorical
from Skin_Project.params import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score


def initialize_model():
    model = Sequential()

    if CLASSIFICATION == 'cat':

     # First convolutional layer
        model.add(layers.Conv2D(filters = 256, kernel_size = (5,5), padding = "same", activation = "relu", input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)))

     # Max pooling layer and BatchNormalization layer
        model.add(layers.MaxPool2D(pool_size = (2,2)))
        #model.add(layers.BatchNormalization())

     # Second convolutional layer
        model.add(layers.Conv2D(filters = 128,
                kernel_size = (3,3),
                padding = "same",
                activation = "relu"))

     # Max pooling layer and BatchNormalization layer
        model.add(layers.MaxPool2D(pool_size = (2,2)))
        #model.add(layers.BatchNormalization())

        # Third convolutional layer
        model.add(layers.Conv2D(filters = 64,
                kernel_size = (3,3),
                padding = "same",
                activation = "relu"))

     # Max pooling layer and BatchNormalization layer
        model.add(layers.MaxPool2D(pool_size = (2,2)))
        #model.add(layers.BatchNormalization())

        # Fourth convolutional layer
        #model.add(layers.Conv2D(filters = 256,
        #           kernel_size = (3,3),
        #           padding = "same",
        #           activation = "relu"))

     # Max pooling layer and BatchNormalization layer
        #model.add(layers.MaxPool2D(pool_size = (2,2)))
        #model.add(layers.BatchNormalization())

        # Flattening layer
        model.add(layers.Flatten())

        # Dense layer
        model.add(layers.Dense(units = 256,
                    activation = "relu"))

        model.add(layers.Dropout(0.5))

        # Output layer
        model.add(layers.Dense(7, activation='softmax'))

        return model

    if CLASSIFICATION == 'binary':

        # First convolutional layer
        model.add(layers.Conv2D(filters = 64, kernel_size = (5,5), padding = "same", activation = "relu", input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)))

        # Max pooling layer
        model.add(layers.MaxPool2D(pool_size = (2,2)))
        # Second convolutional layer
        model.add(layers.Conv2D(filters = 64,
                kernel_size = (3,3),
                padding = "same",
                activation = "relu"))

        # Max pooling layer
        model.add(layers.MaxPool2D(pool_size = (2,2)))

        # Third convolutional layer
        model.add(layers.Conv2D(filters = 32,
                kernel_size = (3,3),
                padding = "same",
                activation = "relu"))

        # Max pooling layer
        model.add(layers.MaxPool2D(pool_size = (2,2)))

        # Fourth convolutional layer
        model.add(layers.Conv2D(filters = 256,
                kernel_size = (3,3),
                padding = "same",
                activation = "relu"))

        # Max pooling layer
        model.add(layers.MaxPool2D(pool_size = (2,2)))


        # Flattening layer
        model.add(layers.Flatten())

        # Dense layer
        model.add(layers.Dense(units = 128,
                    activation = "relu"))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))

        return model













def preproc(df_sample, dx):
    preproc = make_column_transformer(
        (FunctionTransformer(lambda x: x/255., feature_names_out='one-to-one'), list(df_sample.drop(columns=dx).columns.values)),
        remainder ='passthrough')
    return preproc


def preprocess():
    print('coucou')
    df = pd.read_csv(CHEMIN_3, index_col=0)

    #df = categorize(df)
    #print('df categorized')
    #df = drop_columns(df)
    #print('columns dropped')

    if CLASSIFICATION == 'binary':
        df= labelize(df)
        print('labelized ok')

    if SAMPLE_SIZE != 1.0 :
        df = sampler(df)
        print (f'df sampled with a ratio of {df.dx.value_counts()[0]/df.shape[0]}')

    preprocess = preproc(df, 'dx')
    df_processed = pd.DataFrame(preprocess.fit_transform(df), columns = preprocess.get_feature_names_out())

    X_processed = df_processed.drop(columns='remainder__dx')
    y_processed=df_processed['remainder__dx']
    print('data in processing...')

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.33, random_state=42)
    print('data split')

    X_train = np.array(X_train).reshape(len(X_train), IMAGE_SIZE, IMAGE_SIZE, 3)
    X_test = np.array(X_test).reshape(len(X_test), IMAGE_SIZE, IMAGE_SIZE, 3)
    print('data reshaped :)')

    if CLASSIFICATION=='cat':
        ### Encoding the labels
        y_train = to_categorical(y_train, 7)
        y_test = to_categorical(y_test, 7)
        print('')

    return X_train, X_test, y_train, y_test






def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam()

    if CLASSIFICATION=='cat':
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()
                               , tf.keras.metrics.FBetaScore(beta=2.0)])

    elif CLASSIFICATION=='binary':
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()
                               , tf.keras.metrics.FBetaScore(beta=2.0)])

    return model


def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=10,
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
        verbose=1
    )

    return model, history


def evaluate_model(model,X_test,y_test,threshold, batch_size=256):

    if model is None:
        print(f"No model to evaluate")
        return None

    if CLASSIFICATION == 'binary':
        y_pred = model.predict(X_test)
        threshold = THRESHOLD
        y_binary_predictions = (y_pred > threshold).astype(int)
        accuracy = accuracy_score(y_test, y_binary_predictions)
        precision = precision_score(y_test, y_binary_predictions)
        recall = recall_score(y_test, y_binary_predictions)
        f2 = fbeta_score(y_test, y_binary_predictions,beta = 2.0)

        metrics_dict = {'Threshold':threshold,'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F2 Score':f2}
        df_metrics = pd.DataFrame(metrics_dict,index=[threshold])
        return df_metrics

    if CLASSIFICATION == 'cat':

        metrics = model.evaluate(
            x=X_test,
            y=y_test,
            batch_size=batch_size,
            verbose=0,
            # callbacks=None,
            return_dict=True)

        return metrics



def train():
    X_train, X_test, y_train, y_test = preprocess()

    print(X_train.shape)
    print(y_train.shape)
    print(y_train)

    model = initialize_model()
    print('model initialized...')

    model = compile_model(model)
    print('model compiled')

    model, history = train_model(model, X_train,y_train)
    print('model trained!')

    metrics = evaluate_model(model, X_test, y_test,threshold=THRESHOLD)

    #print(f'loss is {metrics["loss"]}')
    #print(f'accuracy is {metrics["accuracy"]}')
    #print(f'recall is {metrics["recall"]}')
    #print(f'precision is {metrics["precision"]}')
    print(metrics)
    return metrics


if __name__ == '__main__':
    train()






import time
import os
import glob

from tensorflow import keras
#import mlflow
from Skin_Project.params import *

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(CHEMIN_4, f"{timestamp}.h5")
    model.save(model_path)
    print("Model saved locally")

    return None



def load_best_model():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":

        if CLASSIFICATION == 'binary':
            # Get the latest model version name by the timestamp on disk
            local_best_model_path = f"{CHEMIN_BINARY}/best_model.h5"

            if not local_best_model_path:
                print('No best model saved yet')
                return None

            best_model = keras.models.load_model(local_best_model_path)

            print("Best model for binary classification loaded from local disk")

            return best_model

        if CLASSIFICATION == 'cat':
            # Get the latest model version name by the timestamp on disk
            local_best_model_path = f"{CHEMIN_CAT}/best_model.h5"

            if not local_best_model_path:
                print('No best model saved yet')
                return None

            best_model = keras.models.load_model(local_best_model_path)

            print("Best model for multiclass classification loaded from local disk")

            return best_model
    pass



def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":

        # Get the latest model version name by the timestamp on disk
        local_model_paths = glob.glob(f"{CHEMIN_4}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print('local_model_paths',local_model_paths)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("Model loaded from local disk")

        return latest_model

    return None
