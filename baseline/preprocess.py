import os
SAMPLE_SIZE = os.environ.get('SAMPLE_SIZE',0.5)

# Labels between 1 (malign) and 0 (benign)
def labelize(df):
    benign_classes = [4,2,3]
    for i in range(1,df.shape[0]+1):
        if df['label'][i-1] in benign_classes:
            df ['label'][i-1] = 0
        else:
            df['label'][i-1] = 1
    return df

#Splits into a SAMPLE_SIZE% sample
def sampler(df):
    df_sample = df.sample(SAMPLE_SIZE*df.shape[0], random_state = 42)
    return df_sample
