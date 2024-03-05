from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
import pandas as pd



def preproc(df):
    preproc = make_column_transformer(
        (FunctionTransformer(lambda x: x/255., feature_names_out='one-to-one'), list(df.drop(columns='label').columns.values)),
        remainder ='passthrough')
    return preproc
