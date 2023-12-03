import joblib
import pickle
import numpy as np

loaded_model = joblib.load('Search/knn_model.joblib')


with open('Search/data_matrix.pkl', 'rb') as file:
    data_matrix = pickle.load(file)

with open('Search/data_pivot.pkl', 'rb') as file:
    data_pivot = pickle.load(file)



query_index = np.random.choice(data_pivot.shape[0])
    #print(n, query_index)
distance, indice = loaded_model.kneighbors(data_pivot.iloc[query_index].values.reshape(1,-1), n_neighbors=6)
# distances, indices = loaded_model.kneighbors([new_data_point])
for i in range(0, len(distance.flatten())):
    if  i == 0:
        print('Recmmendation for ## {0} ##:'.format(data_pivot.index[query_index]))
    else:
        print('{0}: {1} with distance: {2}'.format(i, data_pivot.index[indice.flatten()[i]],
                                                   distance.flatten()[i]))
    # print('\n')