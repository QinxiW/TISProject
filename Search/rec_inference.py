import joblib


loaded_model = joblib.load('Search/knn_model.joblib')

new_data_point = 0 # TODO
distances, indices = loaded_model.kneighbors([new_data_point])