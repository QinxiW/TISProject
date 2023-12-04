"""
This workflow performs data cleaning and processing of the following steps:
Step 1: Create Id
Step 2: Level geographic fields
Step 3: Add currency for price
Step 4: Clean text columns
Step 5: Extract wine year
Step 6: Fill missing data by imputation
Step 7: Remove duplicated values
Step 8: Indicate repeated viewer but unique taster-title to dataframe
Step 9: Find similar winery, province, region, designation and fix typo
Step 10: NLP cleanup reviews
"""
import pandas as pd
import numpy as np
import string
import logging
import nltk
import re
from autocorrect import Speller
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words

nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

logger = logging.Logger('data_clea')

text_columns = ['country', 'description', 'designation', 'province_leveled', 'region_1_leveled', 'region_2_leveled',
                'taster_name', 'taster_twitter_handle', 'title', 'variety', 'winery']


def text_columns_process(wine_df, text_columns=text_columns):
    # Turn all the words to lowercase

    for col in text_columns:
        wine_df['{}_lowercase'.format(col)] = wine_df[col].str.lower()

    # ## Remove whitespace
    for col in text_columns:
        wine_df['{}_strip'.format(col)] = wine_df['{}_lowercase'.format(col)].str.strip()

    # ## Remove special characters and puncutation
    for col in text_columns:
        text = wine_df['{}_strip'.format(col)].str.replace('[^\w]', ' ', regex=True)
        remove_punc = dict((ord(char), None) for char in string.punctuation)
        wine_df['{}_cleaned'.format(col)] = text.str.translate(remove_punc)

    # ## Strip accents
    for col in text_columns:
        text_col = wine_df['{}_cleaned'.format(col)]
        wine_df['{}_cleaned'.format(col)] = text_col.apply(lambda x: unidecode(str(x)))
        wine_df['{}_cleaned'.format(col)] = wine_df['{}_cleaned'.format(col)].str.strip()

    return wine_df


def tokenize_and_lemmatize(wine_df_merged):
    wl = WordNetLemmatizer()
    _speller = Speller(lang='en')
    _non_word_match = re.compile(r'[^\w]+')
    _run_of_digits = re.compile(r'\d+')
    stopws = set(stopwords.words('english'))

    def toki(text):
        return [word for sent in sent_tokenize(text)
                for word in word_tokenize(sent) if word not in stopws]

    tokens = wine_df_merged['description_cleaned'].apply(lambda x: toki(x))
    wine_df_merged['description_cleaned_tokenized'] = tokens

    corrected_tokens = [[_speller(token[0])] for token in tokens if
                        _run_of_digits.search(token[0]) or token[0].isalpha()]

    token_length_min = 1
    token_length_max = 20

    filtered_tokens = [[token for token in sublist if token_length_min < len(token) <= token_length_max] for sublist in
                       tokens]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [[lemmatizer.lemmatize(token) for token in sublist] for sublist in filtered_tokens]

    lemmatized_words_nonu = [[le for le in sublist if not le.isdigit()] for sublist in lemmatized_words]

    lemmatized_words_eng = [[word for word in sublist if word in words.words()] for sublist in lemmatized_words_nonu]
    wine_df_merged['description_lemmatized_eng'] = lemmatized_words_eng

    processed_text = [' '.join(lemmatized_word for lemmatized_word in sublist) for sublist in lemmatized_words_nonu]
    wine_df_merged['description_processed_text'] = processed_text

    return wine_df_merged


def text_facet_openrefine(df):
    # done in openRefine, call an API with the file over
    return df


def level_geo_fields(wine_df):
    wine_df['region_2_leveled'] = np.where(wine_df['region_1'] == wine_df['region_2'], np.nan, wine_df['region_2'])
    wine_df['province_leveled'] = np.where(wine_df['province'] == wine_df['country'], np.nan, wine_df['province'])
    wine_df['region_1_leveled'] = np.where(wine_df['region_1'] == wine_df['province'], np.nan, wine_df['region_1'])
    wine_df['region_1_leveled'] = np.where(wine_df['region_1_leveled'] == wine_df['country'], np.nan,
                                           wine_df['region_1_leveled'])

    return wine_df


def main(file_location_path: str):
    df = pd.read_csv(file_location_path)

    # Step 1: Create Id
    df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

    # Step 2: level geographic fields
    df = level_geo_fields(df)

    # Step 3 Add currency for price, and if country is US set to usd
    df['price_currency'] = np.where(df['country'] == 'US', 'usd', '')

    # Step 4 clean text columns
    df = text_columns_process(df)

    # Step 5 Extract wine year
    extracted_years = df['title_cleaned'].str.extract(r'(?<!\d)(19[3-9]\d{1}|20[0-2]\d|2023)(?!\d)').astype(float)
    df['wine_year'] = extracted_years

    # Step 6 fill missing data by imputation
    df['price_imputated'] = df.groupby(['title_cleaned', 'country_cleaned', 'winery_cleaned'])[
        'price'].fillna(method='ffill')
    df['wine_year_imputated'] = df.groupby(['title_cleaned', 'country_cleaned', 'winery_cleaned'])[
        'wine_year'].fillna(method='ffill')

    columns_to_keep = [col for col in df.columns if '_cleaned' in col or '_imputated' in col] + ['points']
    columns_with_mv = df[columns_to_keep].columns[df[columns_to_keep].isnull().any()]
    # Output the number of missing values of each column
    logger.info('Columns with missing values and count: %s', df[columns_with_mv].isnull().sum().sort_values())

    # Step 7: remove duplicated values
    df = df.drop_duplicates(subset=columns_to_keep)

    # Step 8: Indicate repeated viewer but unique taster-title to dataframe
    duplicates_count_title = df.groupby(['taster_name_cleaned', 'title_cleaned']).size().reset_index(
        name='total_reviews_by_taster_on_title')
    duplicates_count = df.groupby(['taster_name_cleaned', 'title_cleaned', 'points']).size().reset_index(
        name='total_unique_score_by_taster_on_title')

    merged_df = pd.merge(df, duplicates_count_title, on=['taster_name_cleaned', 'title_cleaned'])
    df = pd.merge(merged_df, duplicates_count, on=['taster_name_cleaned', 'title_cleaned', 'points'])

    # Step 9 find similar winery, province, region, designation and fix typo
    df = text_facet_openrefine(df)

    # Step 10: NLP cleanup reviews
    df = tokenize_and_lemmatize(df)

    # Step 11: clean up columns and save cleaned file
    for col in df.columns:
        try:
            df.drop(['{}_lowercase'.format(col)], axis=1, inplace=True)
            df.drop(['{}_strip'.format(col)], axis=1, inplace=True)
        except:
            continue

    df.to_csv("{}-processed-and-cleaned.csv".format(file_location_path))


if __name__ == "__main__":
    """
    For conciseness we did not include the original dataset in the path for git but only local reference
    the dataset used for this project can be found at: https://www.kaggle.com/datasets/zynicide/wine-reviews/data
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_location_path", type=str, help="file location of the csv")

    args = parser.parse_args()
    file_location_path = args.file_location_path
    main(file_location_path)
