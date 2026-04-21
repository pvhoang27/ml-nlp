from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam
from scipy.sparse import coo_matrix
import numpy as np

# Tạo ma trận X (15x15) như b4.py
row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
col = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
data = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
X = coo_matrix((data, (row, col)), shape=(15, 15)).toarray()
def autoencoder(input_unit, hidden_unit):
    model = Sequential()
    model.add(Dense(input_unit, input_shape = (15,), activation = 'relu'))
    model.add(Dense(hidden_unit, activation = 'relu'))
    model.add(Dense(input_unit, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(),
                  metrics = ['accuracy'])
    model.summary()
    return model

model_auto = autoencoder(input_unit = 15, hidden_unit = 6)

model_auto.fit(X, X, epochs = 5, batch_size = 3)

# Phần trích xuất ma trận nhúng
embedding_matrix = model_auto.layers[2].get_weights()[0]
bias = model_auto.layers[2].get_weights()[1]

print('Shape of embedding_matrix: ', embedding_matrix.shape)
print('Embedding_matrix: \n', embedding_matrix)