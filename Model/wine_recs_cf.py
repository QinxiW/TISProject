import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD


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
    print('Recommending ' + str(num) + ' products similar to ' + item(item_id, train_wine) + ' ...')
    print('-----')
    recs = results[item_id][:num]
    for rec in recs:
        print('Recommended: ' + item(rec[1], train_wine) + '(score: ' + f"{rec[0]:.2f}" + ')')


def main():
    df = pd.read_csv('Data/cleaned_data.csv.gz')
    train_wine, test_wine = data_split(wine_df=df)

    tfidf_matrix = build_tfidf_matrix(train_wine=train_wine)

    # First finding the cosine similarities for the tfidf matrix
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Next, appending the results to a dictionary of the similar items to each wine
    results = {}
    for idx, row in train_wine.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:100:-1]
        similar_items = [(cosine_similarities[idx][i], train_wine['wineId'][i]) for i in similar_indices]
        results[row['wineId']] = similar_items[1:]

    # itemId (wineId) is grabbed from the trainset of wines
    itemId = train_wine.loc[:, 'wineId'].values[0]
    itemName = train_wine.loc[train_wine['wineId'] == itemId, 'title'].values[0]

    print(f"Using itemId {itemId} which is {itemName} \n")

    # The recommend function is then run to find and return the top num matches (5 in this case)
    recommend(results=results, item_id=itemId, num=5, train_wine=train_wine)

    # We can then compare the descriptions of the two wines to see how they match
    description_original = train_wine.loc[train_wine['title'] == itemName, 'description'].values[0]

    description_matched = train_wine.loc[train_wine['wineId'] == results[itemId][0][1], 'description'].values[0]

    print(f"First wine description: \n{description_original} \n\nMatched wine description: \n{description_matched}")


if __name__ == '__main__':
    main()