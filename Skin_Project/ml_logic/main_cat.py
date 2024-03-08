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
from Skin_Project.ml_logic.registry import save_model, load_model, load_best_model


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

    if CLASSIFICATION == 'local':
        #Load the best model
        best_model = load_best_model()

        if best_model == None:
            if CLASSIFICATION == 'binary':
                best_model_path = f"{CHEMIN_BINARY}/best_model.h5"
                model.save(best_model_path)
                print("First model is saved as best model !")
                pass
            if CLASSIFICATION == 'cat':
                best_model_path = f"{CHEMIN_CAT}/best_model.h5"
                model.save(best_model_path)
                print("First model is saved as best model !")
                pass

        best_metrics = evaluate_model(best_model, X_test, y_test,threshold=THRESHOLD)
        print(f'ancient metrics are : {best_metrics}')

        metrics = evaluate_model(model, X_test, y_test,threshold=THRESHOLD)
        print(f'new metrics are : {metrics}')

        keys_ =list(metrics.keys())

        if metrics[keys_[2]]>best_metrics['Recall'] and metrics[keys_[1]]>0.5:
            if CLASSIFICATION == 'binary':
                best_model_path = f"{CHEMIN_BINARY}/best_model.h5"
                model.save(best_model_path)
                print("New best model (binary) !")
                return metrics
            if CLASSIFICATION == 'cat':
                best_model_path = f"{CHEMIN_CAT}/best_model.h5"
                model.save(best_model_path)
                print("New best model (multiclass) !")
                return metrics

        else :
            print('The new model is not better than the best model, try again ! :(')
            return best_metrics
    pass


if __name__ == '__main__':
    train()
