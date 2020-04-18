import random

import numpy as np
import pandas as pd


def load_dataset():
    data = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'date'])
    data = np.array(data.iloc[:, :3]).tolist()
    np.random.seed(1234)
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]
    print('load data finished')
    print('total data ', len(data))
    return train_data, test_data, data


def describe_dataset():
    train_data, test_data, data = load_dataset()
    df = pd.DataFrame(data)
    print(df.iloc[:, 2].describe())
