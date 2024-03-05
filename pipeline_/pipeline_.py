from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers, callbacks

from model import initialize_dumb_model, compile_model, train_model, evaluate_model


def preproc(df_sample):
    preproc = make_column_transformer(
        (FunctionTransformer(lambda x: x/255., feature_names_out='one-to-one'), list(df_sample.drop(columns='label').columns.values)),
        remainder ='passthrough')
    return preproc


def preprocessing():
    df_sample = pd.read_csv("/Users/capucinedarteil/code/Capucine-Darteil/Skin_Images/raw_data/sample.csv", index_col=0)
    preprocess = preproc(df_sample)
    df_processed = pd.DataFrame(preprocess.fit_transform(df_sample), columns = preprocess.get_feature_names_out())

    X_processed = df_processed.drop(columns='remainder__label')
    y_processed=df_processed['remainder__label']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.33, random_state=42)

    X_train = np.array(X_train).reshape(len(X_train), 28, 28, 1)
    X_test = np.array(X_test).reshape(len(X_test), 28, 28, 1)

    model = initialize_dumb_model()
    model = compile_model(model)
    model, history = train_model(model, X_train,y_train)

    metrics = evaluate_model(model, X_test,y_test)

    print(f'loss is {metrics["loss"]}')
    print(f'accuracy is {metrics["accuracy"]}')
    print(f'recall is {metrics["recall"]}')
    print(f'precision is {metrics["precision"]}')
    return metrics


if __name__ == '__main__':
    preprocessing()
