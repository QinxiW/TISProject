#!/usr/bin/env python
import pandas as pd
import numpy as np
import string
import logging
import nltk
import re
import time
from rapidfuzz import fuzz, process

# Implement functionality where a user can search, select,
# and retrieve similar / recommended wines

# Choose a threshold for matching (e.g., 80)
threshold = 80

# search_term = 'aroma'
# column_name = 'description_lemmatized_eng'
# result = df[df[column_name].apply(lambda x: process.extractOne(search_term, [x])[1] >= threshold)]
# print(result)


def search_dataframe(user_input, dataframe):
    # Empty DataFrame to store results
    results = pd.DataFrame()

    cols_to_search = ['country_cleaned', 'description_cleaned_tokenized', 'sentiment',
                      'designation_cleaned', 'points', 'price_imputated', 'province_leveled_cleaned',
                        'region', 'taster_name_cleaned',
                    'title_cleaned', 'variety_cleaned', 'winery_cleaned', 'wine_year_imputated']

    # Iterate through columns we index on
    for column in cols_to_search:
        if 'good' in user_input:
            dataframe = dataframe[dataframe.sentiment == 'POS']
        # Use rapidfuzz to calculate similarity score
        similarity_scores = dataframe[column].apply(lambda x: fuzz.partial_ratio(user_input, str(x)))
        # similarity_scores = dataframe[column].apply(lambda x: process.extractOne(user_input, str(x))[1] >= threshold)
        # Filter rows based on a threshold (e.g., 70% similarity)
        matches = dataframe[similarity_scores >= threshold]
        if len(matches):
            print('colum: ', column, 'matches: \n', matches)

        # Add the matches to the results DataFrame
        results = pd.concat([results, matches])

    # Drop duplicates from the results
    results = results.drop_duplicates()

    return results


def main():
    df = pd.read_csv('Data/search.csv.gz')
    df['wine_year_imputated'] = df['wine_year_imputated'].fillna(0).astype(int)
    while True:
        # Get user input
        user_input = input("Search for wine, exit to leave: ")

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        # Search the DataFrame
        results = search_dataframe(user_input, df)

        # Display the results
        print("Matching items:")
        print(results)


if __name__ == '__main__':
    main()