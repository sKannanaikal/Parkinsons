import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('parkinsons.csv')

'''
df.head()
df.info()
df.descibe()
df.columns
'''

# sns.pairplot(df)
# sns.distplot(df['age'])
# df.corr()

data = df.drop('subject#', axis = 1)
data1 = data.drop('age', axis = 1)
data2 = data1.drop('sex', axis = 1)
data3 = data2.drop('test_time', axis = 1)

# print(data3)

X = data3[['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']]
y = data3['total_UPDRS'] # motor_UPDRS

# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

lm = LinearRegression()
lm.fit(X_train, y_train)

prediction = lm.predict(X_test)

plt.scatter(y_test, prediction)
plt.show()

accuracy = lm.score(X_test,y_test)
print(accuracy)

import pickle
#pickle.dump(lm, open('lm_model.pk', 'wb'))
