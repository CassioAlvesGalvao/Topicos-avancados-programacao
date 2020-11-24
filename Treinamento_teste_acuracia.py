from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import numpy as np
import os

# carregando bases de dados de treinamento e de teste(dividi a base 70% / 30%)
dataset_train = numpy.loadtxt("C:\diabetes_data_training.csv", delimiter=",")
dataset_test = numpy.loadtxt("C:\diabetes_data_test.csv", delimiter=",")

# dividindo a base entre as variáveis de treinamento e teste, pesquisar a razão das numerações no data_set
X_train = dataset_train[:,0:16]
Y_train = dataset_train[:,16]

X_test = dataset_test[:,0:16]
Y_test = dataset_test[:,16]

# criação do modelo de camadas #input_dim = dimensões dos dados de entrada
model = Sequential()
model.add(Dense(15, input_dim=16,   kernel_initializer='uniform', activation='relu'     ))
model.add(Dense(12,                 kernel_initializer='uniform', activation='relu'     ))
model.add(Dense(1,                  kernel_initializer='uniform', activation='sigmoid'  ))

# compilando modelo de camadas
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# treinando a rede
model.fit(X_train, Y_train, epochs=400, batch_size=20, verbose=0) #resultado melhorou quando aumentei o numero de epoch para 400

# avaliando a rede com as variáveis de teste
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Model salvo!")


