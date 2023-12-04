"""
This workflow prepares data, trains and evaluate a random forrest model for wine recommendation:
Step 1: Split the data into train and test set
Step 2: Train a random forest classifier model based on training data
Step 3: Evaluate the model trained for metrics include precision, recall, confusion matrix, accuracy and classification report
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score


def data_split(df, label, test_size=0.2, random_state=42):
    """
    prepares train and eval data. Break original df into train and test.
    :param df: the original dataframe
    :param label: the label we set for y
    :param test_size: percentage of the original dataset as test_set
    :param random_state: seed val
    :return: X_train, y_train, X_test, y_test
    """

    # Skip some columns that are either uninformative or holds the answer keys we try to draw
    columns_to_exclude = ['points', 'id']

    # selected_columns = df.loc[:, ~df.columns.isin(columns_to_exclude)]
    numeric_columns = df.select_dtypes(exclude=['object'])
    selected_columns = numeric_columns.loc[:, ~numeric_columns.columns.isin(columns_to_exclude)]

    return train_test_split(selected_columns, label, test_size=test_size,
                            random_state=random_state)


def model_train(X_train, y_train, n_estimators=300, random_state=42):
    # Initialize the Random Forest classifier
    random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Train the model on the training data
    random_forest.fit(X_train.fillna(0), y_train)

    return random_forest


def model_eval(model, X_test, y_test):
    # Make predictions on the testing data
    y_pred = model.predict(X_test.fillna(0))

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)


def main():
    print("Loading df from Data/")
    df = pd.read_csv("Data/cleaned_data.csv.gz")
    print("df head: ", df.head())

    X_train, X_test, y_train, y_test = data_split(df=df, label=df.points > 92)

    random_forest = model_train(X_train=X_train, y_train=y_train)

    model_eval(model=random_forest, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    main()
