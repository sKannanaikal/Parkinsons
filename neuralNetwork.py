from __future__ import print_function
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pickle
import pandas as pd
import numpy as np
import keras

column_names = ["subject#" ,"age" ,"sex" ,"test_time"
,"motor_UPDRS","total_UPDRS","Jitter(%)","Jitter(Abs)",
"Jitter:RAP","Jitter:PPQ5","Jitter:DDP","Shimmer","Shimmer(dB)"
,"Shimmer:APQ3","Shimmer:APQ5","Shimmer:APQ11","Shimmer:DDA",
"NHR","HNR","RPDE","DFA","PPE"]

data = pd.read_csv("parkinsons.csv")

df = pd.DataFrame(data, columns = column_names)

data = df.drop('subject#', axis = 1)
data1 = data.drop('age', axis = 1)
data2 = data1.drop('sex', axis = 1)
data3 = data.drop('test_time', axis = 1)

X = data3[["motor_UPDRS", "Jitter(%)","Jitter(Abs)",
"Jitter:RAP","Jitter:PPQ5","Jitter:DDP","Shimmer","Shimmer(dB)"
,"Shimmer:APQ3","Shimmer:APQ5","Shimmer:APQ11","Shimmer:DDA",
"NHR","HNR","RPDE","DFA","PPE"]]

y = data3[["total_UPDRS"]]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def get_model():
    
    model = Sequential()
    model.add(Dense(20, input_dim=17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model()

batch_size = 1000
epochs = 10000

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
predictions = model.predict(x_test)


roundedModels = np.round(predictions, 0)
properPredictions = []
for value in roundedModels:
    x = int(value)
    properPredictions.append(x)


print(properPredictions)
print(y_test)
#print(accuracy_score(properPredictions, y_test))

pickle.dump(model, open('neuralNetwork_model.pk', 'wb'))

