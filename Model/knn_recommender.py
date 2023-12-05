"""
This tasks creates a recommender system using k-nearest neighbor model:
Step 1: Loads the dataframe and split into train and test sets, create user and item id
Step 2: Build the data pivot frame and matrix using info about the wine and sentiment of the review
Step 3: Train a k-nearest neighbor model using the data matrix built above
Step 4: Adhoc test that allows query for random index and get recs back
Step 5: Save the final models and artifacts to output files
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib
import pickle


def build_data(df):
    data_recommend = df[['province_leveled_cleaned', 'variety_cleaned', 'sentiment_score']]
    data_recommend.dropna(axis=0, inplace=True)
    data_recommend.drop_duplicates(['province_leveled_cleaned', 'variety_cleaned'], inplace=True)

    data_pivot = data_recommend.pivot(index='variety_cleaned', columns='province_leveled_cleaned',
                                      values='sentiment_score').fillna(0)
    data_matrix = csr_matrix(data_pivot)

    with open('data_matrix.pkl', 'wb') as file:
        pickle.dump(data_matrix, file)
    with open('data_pivot.pkl', 'wb') as file:
        pickle.dump(data_pivot, file)
    return data_matrix, data_pivot


def build_model(data_matrix):
    knn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
    model_knn = knn.fit(data_matrix)

    joblib.dump(model_knn, 'Search/knn_model.joblib')

    return model_knn


def main():
    df = pd.read_csv('Data/search.csv.gz')
    data_matrix, data_pivot = build_data(df)

    model_knn = build_model(data_matrix)

    # run a mini test by picking a random a query_index from the data poviot and check recs back
    for _ in range(5):
        query_index = np.random.choice(data_pivot.shape[0])
        # print(n, query_index)
        distance, indice = model_knn.kneighbors(data_pivot.iloc[query_index].values.reshape(1, -1), n_neighbors=6)
        for i in range(0, len(distance.flatten())):
            if i == 0:
                print('Got it! Here are similar ones for {0}:'.format(data_pivot.index[query_index]))
            else:
                print('{0}: {1} with distance: {2}'.format(i, data_pivot.index[indice.flatten()[i]],
                                                           distance.flatten()[i]))
        print('\n')


if __name__ == '__main__':
    main()
