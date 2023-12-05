"""
This tasks creates a recommender system by collaborative filtering:
Step 1: Loads the dataframe and split into train and test sets, create user and item id
Step 2: Build the tfidf_matrix using the train set for the wine descriptions
Step 3: Create cosine_similarities for each pairs of wine and review
Step 4: Combine other relevant features with text review data
Step 5: Save the final models and artifacts to output files
"""
import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

import joblib
import pickle


def data_split(wine_df):
    wine_df.loc[:, 'wineId'] = wine_df.loc[:, 'title'].astype('category').cat.codes

    wine_df.loc[:, ['description_cleaned_tokenized', 'wineId', 'title']].head()

    train_wine, test_wine = train_test_split(wine_df, train_size=0.05)

    train_wine.reset_index(drop=True, inplace=True)

    print(f"Training on {len(train_wine)} samples.")
    return train_wine, test_wine


def build_tfidf_matrix(train_wine):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0, stop_words='english')

    tfidf_matrix = tf.fit_transform(train_wine['description_cleaned_tokenized'])

    print(f"The term-frequency inverse document frequency matrix is {tfidf_matrix.shape[0]} by {tfidf_matrix.shape[1]}")

    return tfidf_matrix


def item(id, train_wine):
    return train_wine.loc[train_wine['wineId'] == id]['title'].tolist()[0].split(' - ')[0]


def recommend(train_wine, item_id, num, results):
    print('Recommending ' + str(num) + ' wines similar to ' + item(item_id, train_wine) + ' ...')
    print('-----')
    recs = results[item_id][:num]
    for rec in recs:
        print('Recommended: ' + item(rec[1], train_wine) + '(score: ' + f"{rec[0]:.2f}" + ')')


def combine_text_and_numeric(df):
    variety_description = df[["variety_cleaned", "description_cleaned_tokenized"]]
    variety_description = variety_description.drop_duplicates().dropna()
    variety_rev_number = variety_description["variety_cleaned"].value_counts()

    # Convert the Series to Dataframe
    df_rev_number = pd.DataFrame({'variety': variety_rev_number.index, 'rev_number': variety_rev_number.values})
    # print(df_rev_number[(df_rev_number["rev_number"] > 1)].shape)

    # Create a ist of grape varieties that have more than one review
    variety_multi_reviews = df_rev_number[(df_rev_number["rev_number"] > 1)]["variety"].tolist()
    variety_description_2 = pd.DataFrame(columns=["variety", "description"])

    # Define a CountVectorizer object
    cv = CountVectorizer(stop_words="english", ngram_range=(2, 2))

    # Define a TfidfTransformer object
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

    variety_description = variety_description.set_index('variety_cleaned')

    for grape in variety_multi_reviews:
        df = variety_description.loc[[grape]]

        # Generate word counts for the words used in the reviews of a specific grape variety
        word_count_vector = cv.fit_transform(df["description"])

        # Compute the IDF values
        tfidf_transformer.fit(word_count_vector)

        # Obtain top 100 common words (meaning low IDF values) used in the reviews. Put the IDF values in a DataFrame
        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
        df_idf.sort_values(by=["idf_weights"], inplace=True)

        # Collect top 100 common words in a list
        common_words = df_idf.iloc[:100].index.tolist()

        # Convert the list to a string and create a dataframe
        common_words_str = ", ".join(elem for elem in common_words)
        new_row = {"variety": grape, "description": common_words_str}

        # Add the variety and its common review words to a new dataframe
        variety_description_2 = variety_description_2.append(new_row, ignore_index=True)

    # Define a TfidVectorizer object.
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(2, 2))

    # Count the words in each description, calculate idf, and multiply idf by tf.
    tfidf_matrix = tfidf.fit_transform(variety_description_2["description"])

    # Resulting matrix should be # of descriptions (row) x # of bigrams (column)
    print(tfidf_matrix.shape)

    # Compute the cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a Series, where the index is the grape variety and the element is the index of the wine in the dataset.
    variety_description_2 = variety_description_2.reset_index()
    indices = pd.Series(variety_description_2.index, index=variety_description_2['variety'])

    with open('variety_description_2.pkl', 'wb') as file:
        pickle.dump(variety_description_2, file)

    with open('variety_multi_reviews.pkl', 'wb') as file:
        pickle.dump(variety_multi_reviews, file)

    with open('indices.pkl', 'wb') as file:
        pickle.dump(indices, file)

    with open('cosine_sim.pkl', 'wb') as file:
        pickle.dump(cosine_sim, file)

    return


def main():
    df = pd.read_csv('Data/cleaned_data.csv.gz')
    train_wine, test_wine = data_split(wine_df=df)

    tfidf_matrix = build_tfidf_matrix(train_wine=train_wine)

    # find the cosine similarities for the tfidf matrix
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # add results to a dictionary of the similar items to each wine
    results = {}
    for idx, row in train_wine.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:100:-1]
        similar_items = [(cosine_similarities[idx][i], train_wine['wineId'][i]) for i in similar_indices]
        results[row['wineId']] = similar_items[1:]

    # run a mini test by graping a wineId from the df and get
    itemId = train_wine.loc[:, 'wineId'].values[0]
    itemName = train_wine.loc[train_wine['wineId'] == itemId, 'title'].values[0]
    print(f"Using itemId: {itemId} with name: {itemName} \n")

    # call recommend function to find and return the top num matches
    recommend(results=results, item_id=itemId, num=5, train_wine=train_wine)

    # compare the descriptions of the two wines to see how they match
    description_searched = train_wine.loc[train_wine['title'] == itemName, 'description'].values[0]
    description_matched = train_wine.loc[train_wine['wineId'] == results[itemId][0][1], 'description'].values[0]

    print(
        f"Interacted wine description: \n{description_searched} \n\nMatched wine description: \n{description_matched}")

    combine_text_and_numeric(df)


if __name__ == '__main__':
    main()
