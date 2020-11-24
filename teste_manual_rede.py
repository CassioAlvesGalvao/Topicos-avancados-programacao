from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import numpy as np
import os
import tensorflow as tf

# Criando e carregando o modelo saldo em disco
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# carregando os pesos do modelo
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# compilando o modelo recuperado
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

predictions = loaded_model.predict([[48,1,1,1,1,0,1,1,0,0,0,1,0,0,0,0]]) #aqui adiciono os valores de entrada do usuário, por enquanto fixo no código para teste
print("predictions shape:", predictions.shape)

classes = tf.math.argmax(predictions, axis = 1)
print(classes)
