import numpy as np
import pandas as pd
import seaborn as sns
import pickle 
import matplotlib.pyplot as plt 

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR



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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 1)


model = SVR(gamma= 'scale', C=1.0, epsilon=0.2)

model.fit(X_train, y_train)



predictions = model.predict(X_test)

accuracy = model.score(X_test,y_test)

print(accuracy)
print(predictions)

pickle.dump(model, open('svm_model.pk', 'wb'))