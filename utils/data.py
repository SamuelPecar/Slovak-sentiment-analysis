import pandas as pd
from gensim.models import KeyedVectors

def remap_data_to_2(data):
    for i, sample in enumerate(data):

        if sample[2] > 0:
            data[i][2] = 1
        else:
            data[i][2] = 0
    return data


def remap_data_to_3(data):
    for i, sample in enumerate(data):

        if sample[2] < -1:
            data[i][2] = 2
        elif sample[2] > 1:
            data[i][2] = 1
        else:
            data[i][2] = 0
    return data

def remap_data_to_3_from_text(data):
    for i, sample in enumerate(data):

        if sample[2] == 'Positive':
            data[i][2] = 2
        elif sample[2] == 'Negative':
            data[i][2] = 0
        else:
            data[i][2] = 1
    return data

def save_predictions(name, predictions, original_data):
    # original_data[:, 3] = predictions
    dataframe = pd.DataFrame(data=original_data)
    dataframe['predictions'] = predictions
    dataframe.to_csv(name + '.csv',sep=',', header=False, index=False, quoting=2)


def load_vectors(vector_path, binary=True):
    return KeyedVectors.load_word2vec_format(vector_path, binary=binary)

def get_histogram_data(set):
    histogram = {}

    for item in set:
        if item in histogram:
            histogram[item] += 1
        else:
            histogram[item] = 1
    return histogram