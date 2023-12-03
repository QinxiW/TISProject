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

# print("data_pivot", data_pivot.columns)
# distance, indice = loaded_model.kneighbors(data_pivot.iloc[query_index].values.reshape(1, -1), n_neighbors=6)
# or
# distance, indice = loaded_model.kneighbors(
#     data_pivot[data_pivot.index.get_level_values('variety') == variety].values.reshape(1, -1), n_neighbors=6)


def recommend(varieties: list):
    for variety in varieties[:5]:
        print("variety ", variety)
        # variety = variety.title()
        query_index = data_pivot.index.get_loc(variety)
        # distance, indice = loaded_model.kneighbors(data_pivot[data_pivot.index.get_level_values('variety') == variety].values.reshape(1, -1), n_neighbors=6)
        distance, indice = loaded_model.kneighbors(data_pivot.iloc[query_index].values.reshape(1, -1), n_neighbors=6)
        for i in range(0, len(distance.flatten())):
            if i == 0:
                print('Recmmendation for ## {0} ##:'.format(data_pivot.index[query_index]))
            else:
                # convert back to wine title
                wines = wine_df[wine_df['variety_cleaned'] == variety]
                print('{0}: {1} with distance: {2}'.format(i, data_pivot.index[indice.flatten()[i]],
                                                           distance.flatten()[i]))
                print('wines recommended for you: ', wines[['title_cleaned']][:5])
        print('\n')

def main(variety='Sherry'):
    recommend(variety)


if __name__ == '__main__':
    main()