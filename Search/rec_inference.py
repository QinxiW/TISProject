"""
This tasks runs inference for the knn model for similar wine based on wine attributes and sentiments of the review
The main function will be invoked by the search wine function, and pass in the matched wine attributes
from the user, then recommender then works to find similar wines based on search match
"""

import joblib
import pickle
import numpy as np
import pandas as pd
from rec_comment_sim import rec_and_sim_comment, single_rec

loaded_model = joblib.load('Search/knn_model.joblib')

with open('Search/data_matrix.pkl', 'rb') as file:
    data_matrix = pickle.load(file)

with open('Search/data_pivot.pkl', 'rb') as file:
    data_pivot = pickle.load(file)

wine_df = pd.read_csv('Data/cleaned_data.csv.gz')
wine_df['price'].fillna(0, inplace=True)

def recommend_single(variety):
    query_index = data_pivot.index.get_loc(variety)
    # distance, indice = loaded_model.kneighbors(data_pivot[data_pivot.index.get_level_values('variety') == variety].values.reshape(1, -1), n_neighbors=6)
    distance, indice = loaded_model.kneighbors(data_pivot.iloc[query_index].values.reshape(1, -1), n_neighbors=6)
    for i in range(0, len(distance.flatten())):
        if i == 0:
            # print('Recmmendation for ## {0} ##:'.format(data_pivot.index[query_index]))
            wines = wine_df[wine_df['variety_cleaned'] == variety]
            print('For other wines similar to your search, we also recommend: \n')
            for i in range(3):
                titles = wines.title.tolist()
                prices = wines.price.tolist()
                countries = wines.country.tolist()
                if prices[i]:
                    price = " $" + str(int(prices[i]))
                else:
                    price = ""
                print(i+1, ": ", titles[i] + " from " + countries[i] + "-" + price)
            print("\n")
        else:
            # convert back to wine title
            # wines = wine_df[wine_df['variety_cleaned'] == variety]
            percent_score = "{:.0%}".format(1 - distance.flatten()[i])
            sim_var = data_pivot.index[indice.flatten()[i]]
            print('Here are some wine variety we think you will like: {0}, with recommendation score: {1}'.format(
                sim_var,
                percent_score))

            comm = single_rec(sim_var)['common_words'][:6].tolist()
            if comm:
                print('We found these top common words for the matched wine reviews, if you find anything you like you can search more with them:')
                print(str(comm)[1:-1])
                break
            # print('Similar wine recommended for you: ', wines.title_cleaned.tolist()[:3], 'with')
            # print('\n')


def recommend(varieties: list):
    for variety in varieties[:5]:
        # print("variety ", variety)
        # variety = variety.title()
        recommend_single(variety)


def main(variety='sherry'):
    recommend_single(variety)


if __name__ == '__main__':
    main()
