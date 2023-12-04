import pandas as pd
import numpy as np
import ast


def main():
    df = pd.read_csv('Data/sentiment_data.csv.gz')
    df['base_sentiment_label_dict'] = df['base_sentiment_label'].apply(lambda x: ast.literal_eval(x.strip()))

    print("df['base_sentiment_label_dict'].head(): ", df['base_sentiment_label_dict'].head())

    df[['sentiment', 'sentiment_score']] = df['base_sentiment_label_dict'].apply(
        lambda x: pd.Series([x['label'], x['score']]))

    print(df.head())

    # Index(['Unnamed: 0', 'id', 'country', 'description', 'designation', 'points',
    #        'price', 'province', 'region_1', 'region_2', 'taster_name',
    #        'taster_twitter_handle', 'title', 'variety', 'winery',
    #        'region_2_leveled', 'province_leveled', 'region_1_leveled',
    #        'price_currency', 'country_cleaned', 'description_cleaned',
    #        'designation_cleaned', 'province_leveled_cleaned',
    #        'region_1_leveled_cleaned', 'region_2_leveled_cleaned',
    #        'taster_name_cleaned', 'taster_twitter_handle_cleaned', 'title_cleaned',
    #        'variety_cleaned', 'winery_cleaned', 'wine_year', 'price_imputated',
    #        'wine_year_imputated', 'total_reviews_by_taster_on_title',
    #        'total_unique_score_by_taster_on_title',
    #        'description_cleaned_tokenized', 'description_lemmatized_eng',
    #        'description_processed_text'],
    #       dtype='object')

    df['region'] = df['region_1_leveled'].fillna(df['region_2_leveled'])

    cols_to_keep = ['country_cleaned', 'province_leveled_cleaned', 'designation_cleaned',
                    'description_cleaned_tokenized',
                    'sentiment', 'sentiment_score', 'points', 'price_imputated',
                    'region', 'taster_name_cleaned',
                    'title_cleaned', 'variety_cleaned', 'winery_cleaned', 'wine_year_imputated']

    df[cols_to_keep].to_csv('Data/search.csv')


if __name__ == '__main__':
    main()
