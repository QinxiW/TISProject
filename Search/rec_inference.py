"""
This tasks runs inference for the knn model for similar wine based on wine attributes and sentiments of the review
The main function will be invoked by the search wine function, and pass in the matched wine attributes
from the user, then recommender then works to find similar wines based on search match
"""

import joblib
import pickle
import numpy as np
import pandas as pd

loaded_model = joblib.load('Search/knn_model.joblib')

with open('Search/data_matrix.pkl', 'rb') as file:
    data_matrix = pickle.load(file)

with open('Search/data_pivot.pkl', 'rb') as file:
    data_pivot = pickle.load(file)

wine_df = pd.read_csv('Data/search.csv.gz')


def recommend_single(variety):
    query_index = data_pivot.index.get_loc(variety)
    # distance, indice = loaded_model.kneighbors(data_pivot[data_pivot.index.get_level_values('variety') == variety].values.reshape(1, -1), n_neighbors=6)
    distance, indice = loaded_model.kneighbors(data_pivot.iloc[query_index].values.reshape(1, -1), n_neighbors=6)
    for i in range(0, len(distance.flatten())):
        if i == 0:
            # print('Recmmendation for ## {0} ##:'.format(data_pivot.index[query_index]))
            print('Finding wines similar to your search..')
        else:
            # convert back to wine title
            wines = wine_df[wine_df['variety_cleaned'] == variety]
            print('{0}: {1} with distance: {2}'.format(i, data_pivot.index[indice.flatten()[i]],
                                                       distance.flatten()[i]))
            print('wines recommended for you: ', wines.title_cleaned.tolist()[:3])
            print('\n')


def recommend(varieties: list):
    for variety in varieties[:5]:
        # print("variety ", variety)
        # variety = variety.title()
        recommend_single(variety)


def main(variety='sherry'):
    recommend_single(variety)


if __name__ == '__main__':
    main()
