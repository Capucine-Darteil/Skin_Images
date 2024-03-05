import os
import get_data, merge_dicts, resize_data, flat_images, df_final

CHEMIN_1 = os.environ.get('CHEMIN_1')
CHEMIN_2 = os.environ.get('CHEMIN_2')
CHEMIN_METADATA = os.environ.get('CHEMIN_METADATA')

def load_data(CHEMIN_1):
    dict_1 = get_data(CHEMIN_1)
    dict_2 = get_data(CHEMIN_2)
    data = merge_dicts(dict_1, dict_2)
    return data

def preprocess_data(data):
    data = resize_data(data)
    df = flat_images(data)
    df = df_final(CHEMIN_METADATA,df)
    return df
