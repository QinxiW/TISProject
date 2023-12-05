"""
This tasks runs inference for the cf model using tfidf matrix for similar wine and review matching
The main function will be invoked by the search wine function, and pass in the matched wine attributes
from the user, then recommender then works to find similar wines based on search match and also draw
out the bag of words in the review that are shared between the similar wines
"""

import joblib
import pickle
import numpy as np
import pandas as pd

# Load the trained artifacts
with open('Search/cosine_sim.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

with open('Search/indices.pkl', 'rb') as file:
    indices = pickle.load(file)

with open('Search/variety_description_2.pkl', 'rb') as file:
    variety_description_2 = pickle.load(file)

with open('Search/variety_multi_reviews.pkl', 'rb') as file:
    variety_multi_reviews = pickle.load(file)


def rec_and_sim_comment(var_list):
    """
    Higher level function that handles the interfacing of other caller,
    and run single_rec by looping through the item in matched list
    :param var_list: matched wine list
    :return: None
    """
    for variety in var_list:
        print(single_rec(variety))
    return


def single_rec(variety, cosine_sim=cosine_sim):
    # Get the index of the input wine
    if variety not in indices:
        return

    idx = indices[variety]

    # pairwise similarity scores between the input wine and all the wines
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort base on the scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # grab top 3 highest similar scores
    sim_scores = sim_scores[1:4]

    # Get the grape variety indices
    wine_idx_list = [i[0] for i in sim_scores]

    # Create the output dataframe
    df = pd.DataFrame(columns=["similar wines", "Top 6 common words in wine reviews"])

    for wine_idx in wine_idx_list:
        review_variety = variety_description_2.iloc[wine_idx]["variety"]

        # Get top 6 common words in the review
        des = variety_description_2.iloc[wine_idx]["description"]

        if review_variety in variety_multi_reviews:  # If the wine has more than one reviews
            des_split = des.split(", ")
            key_words_list = des_split[:6]
            key_words_str = ", ".join(key_words_list)

        else:
            key_words_str = des

        new_row = {"similar wines": review_variety, "Top 6 common words in wine reviews": key_words_str}
        df = df._append(new_row, ignore_index=True)

    df.set_index("similar wines")

    # Widen the column width so that all common words could be displayed
    pd.set_option('max_colwidth', 500)

    return df


def main(variety='sherry'):
    rec_and_sim_comment(variety)


if __name__ == '__main__':
    main()
