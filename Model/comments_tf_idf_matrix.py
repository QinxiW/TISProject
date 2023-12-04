import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('Data/cleaned_data.csv.gz')

variety_description = df[["variety_cleaned", "description_cleaned_tokenized"]]
variety_description = variety_description.drop_duplicates().dropna()
# Count the number of reviews per grape variety. This returns a series.
variety_rev_number = variety_description["variety_cleaned"].value_counts()

# Convert the Series to Dataframe
df_rev_number = pd.DataFrame({'variety': variety_rev_number.index, 'rev_number': variety_rev_number.values})
print(df_rev_number[(df_rev_number["rev_number"] > 1)].shape)

# Create a ist of grape varieties that have more than one review
variety_multi_reviews = df_rev_number[(df_rev_number["rev_number"] > 1)]["variety"].tolist()

# Create a ist of grape varieties that have only one review
variety_one_review = df_rev_number[(df_rev_number["rev_number"] == 1)]["variety"].tolist()
variety_description_2 = pd.DataFrame(columns=["variety", "description"])

# Define a CountVectorizer object
cv = CountVectorizer(stop_words="english", ngram_range=(2, 2))

# Define a TfidfTransformer object
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

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

variety_description = variety_description.set_index('variety_cleaned')
variety_description_2 = pd.DataFrame(columns=["variety", "description"])

with open('variety_description_2.pkl', 'wb') as file:
    pickle.dump(variety_description_2, file)

with open('variety_multi_reviews.pkl', 'wb') as file:
    pickle.dump(variety_multi_reviews, file)

with open('indices.pkl', 'wb') as file:
    pickle.dump(indices, file)

with open('cosine_sim.pkl', 'wb') as file:
    pickle.dump(cosine_sim, file)
