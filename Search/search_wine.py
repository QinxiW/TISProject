#!/usr/bin/env python
import pandas as pd
import numpy as np
import string
import logging
import nltk
import re
import time
from rapidfuzz import fuzz

search_term = 'aroma'
column_name = 'description_lemmatized_eng'

# Choose a threshold for matching (e.g., 80)
threshold = 0

df = pd.read_csv('Data/cleaned_data.csv.gz')
# print(df.head)

# result = df[df[column_name].apply(lambda x: process.extractOne(search_term, [x])[1] >= threshold)]
#
# print(result)

def search_dataframe(user_input, dataframe):
    # Empty DataFrame to store results
    results = pd.DataFrame()

    # Iterate through each column
    for column in dataframe.columns:
        # Use rapidfuzz to calculate similarity score
        similarity_scores = dataframe[column].apply(lambda x: fuzz.partial_ratio(user_input, str(x)))

        # Filter rows based on a threshold (e.g., 70% similarity)
        matches = dataframe[similarity_scores >= 70]

        # Add the matches to the results DataFrame
        results = pd.concat([results, matches])

    # Drop duplicates from the results
    results = results.drop_duplicates()

    return results

# Get user input
user_input = input("Enter a word or short phrase: ")

# Search the DataFrame
results = search_dataframe(user_input, df)

# Display the results
print("Matching items:")
print(results)
