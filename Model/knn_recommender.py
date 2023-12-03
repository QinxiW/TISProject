import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def build_data(df):
    data_recommend = df[['province', 'variety', 'points']]
    data_recommend.dropna(axis=0, inplace=True)
    data_recommend.drop_duplicates(['province', 'variety'], inplace=True)

    data_pivot = data_recommend.pivot(index='variety', columns='province', values='points').fillna(0)
    data_matrix = csr_matrix(data_pivot)
    return data_matrix, data_pivot


def build_model(data_matrix):
    knn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
    model_knn = knn.fit(data_matrix)

    return model_knn


def main():
    df = pd.read_csv('Data/cleaned_data.csv.gz')
    data_matrix, data_pivot = build_data(df)

    model_knn = build_model(data_matrix)

    for n in range(5):
        query_index = np.random.choice(data_pivot.shape[0])
        # print(n, query_index)
        distance, indice = model_knn.kneighbors(data_pivot.iloc[query_index].values.reshape(1, -1), n_neighbors=6)
        for i in range(0, len(distance.flatten())):
            if i == 0:
                print('Recmmendation for ## {0} ##:'.format(data_pivot.index[query_index]))
            else:
                print('{0}: {1} with distance: {2}'.format(i, data_pivot.index[indice.flatten()[i]],
                                                           distance.flatten()[i]))
        print('\n')


if __name__ == '__main__':
    main()