import numpy as np
import pandas as pd


def read_data(src_data='data/pos.txt'):
    # Creating labels for positive data
    if 'pos' in src_data:
        tmp_label = np.full((2500,), 1)
    if 'neg' in src_data:
        tmp_label = np.full((2500,), 0)
    df_label = pd.DataFrame(data=tmp_label, columns=['label'])

    # Reading data
    df_text = pd.read_csv(src_data, header=None, names=['text'])
    df_text_label = pd.concat((df_text, df_label), axis=1)

    return df_text_label


def write_predicted_data(df, path='data/', filename='predicted_eval.csv'):
    df.to_csv(path+filename)
