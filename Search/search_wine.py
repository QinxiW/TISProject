"""
This is the main task where a user can search, select,and retrieve similar / recommended wines.
The functionality is built using interactive command line input exchange, where a user can search
in english words or short phrases what type of wine (e.g 'sherry', 'italian') or wine attributes
they want (e.g 'aromatic', 'fruit flavors'). These are then used as search phrases against our
database, the top retrieval quality wines are then anchored to use for additional recommendation
inference, and added the final results matched for the user: 'we found these wines for you based
on your search, similar wines related to your search also include...'
"""
#!/usr/bin/env python
import pandas as pd
import numpy as np
import string
import logging
import nltk
import re
import time
from rapidfuzz import fuzz, process
from rec_inference import recommend, recommend_single

# Implement functionality where a user can search, select,
# and retrieve similar / recommended wines

# threshold for matching [0, 100]
threshold = 96

# search_term = 'aroma'
# column_name = 'description_lemmatized_eng'
# result = df[df[column_name].apply(lambda x: process.extractOne(search_term, [x])[1] >= threshold)]
# print(result)


def search_dataframe(user_input, dataframe):
    # Empty DataFrame to store results
    results = pd.DataFrame()

    # cols_to_search = ['country_cleaned', 'description_cleaned_tokenized', 'sentiment',
    #                   'designation_cleaned', 'points', 'price_imputated', 'province_leveled_cleaned',
    #                     'region', 'taster_name_cleaned',
    #                 'title_cleaned', 'variety_cleaned', 'winery_cleaned', 'wine_year_imputated']
    cols_to_search = ['combined']

    # Iterate through columns we index on
    for column in cols_to_search:
        # if 'good' in user_input or 'great' in user_input:
        #     dataframe = dataframe[dataframe.sentiment == 'POS']
        # Use rapidfuzz to calculate similarity score
        similarity_scores = dataframe[column].apply(lambda x: fuzz.partial_ratio(user_input, str(x)))
        # print('similarity_scores', similarity_scores)
        # similarity_scores = dataframe[column].apply(lambda x: process.extractOne(user_input, str(x))[1] >= threshold)

        # Filter rows based on a threshold (e.g., 70% similarity)
        matches = dataframe[similarity_scores >= threshold]

        result_df = dataframe.copy()
        result_df['similarity_scores'] = similarity_scores

        # Sort the DataFrame based on similarity scores in descending order
        result_df = result_df.sort_values(by='similarity_scores', ascending=False)

        # Filter the top 3 rows
        top3_matches = result_df.head(3)
        # if len(top3_matches):
        #     print('colum: ', column, 'matches: \n', matches)

        # Add the matches to the results DataFrame
        results = pd.concat([results, top3_matches])

    # Drop duplicates from the results
    results = results.drop_duplicates()
    return results


def main():
    print('Welcome to the wine community! We hope you will find wine you enjoy here via search and recommendation.')
    df = pd.read_csv('Data/cleaned_data.csv.gz')
    comb_rows = ['country_cleaned', 'province_leveled_cleaned', 'designation_cleaned',
                    'description_cleaned_tokenized', 'points', 'price_imputated',
                    'taster_name_cleaned', 'title_cleaned', 'variety_cleaned', 'winery_cleaned', 'wine_year_imputated']
    # df['wine_year_imputated'] = df['wine_year_imputated'].fillna(0).astype(int)
    print("Let's start by telling me about what wine you are looking for. Type exit to leave anytime.")
    df['combined'] = df[comb_rows].apply(lambda row: ' '.join(map(str, row)), axis=1)
    while True:
        # Get user input
        user_input = input("you can search with a short phrases what type of wine (e.g 'sherry', 'italian') "
                           "or wine attributes (e.g 'aromatic', 'fruit flavors'): ")

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        # Search the DataFrame
        user_input = user_input.lower()
        results = search_dataframe(user_input, df[['combined', 'variety_cleaned', 'title', 'country', 'price']])
        results['price'].fillna(0, inplace=True)
        print("\nGot it!\n")

        # Display the results
        print("We found the following wine title based on your search:\n")
        for i in range(len(results)):
            titles = results.title.tolist()
            prices = results.price.tolist()
            countries = results.country.tolist()
            if prices[i]:
                price = " $" + str(int(prices[i]))
            else:
                price = ""
            print(i+1, ": ", titles[i] + " from " + countries[i] + "-" + price)
        print("\n")
        # print(results[['variety_cleaned', 'province_leveled_cleaned']])

        time.sleep(1)
        # call rec inference for wine similarity and comments similarity
        for variety in results['variety_cleaned'].tolist():
            recommend_single(variety)
            print("\n")
            break


if __name__ == '__main__':
    main()