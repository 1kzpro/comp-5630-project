from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import sys

df = pd.read_csv("data/bank-full.csv", sep=";")

print(df)

enc = OrdinalEncoder()

X_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

enc.fit(X_train)
enc.transform()

sys.exit()
y_train = df.iloc[:,-1]

# print(X_train)
# print(y_train)

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X_train, y_train)

df2 = pd.read_csv("data/bank-additional.csv", sep=";")
X_test = df2.iloc[:,:-1]
y_test = df2.iloc[:,-1]

y_predict = clf.predict(X_test)

accuracy = clf.score(X_test, y_test)

print("Accuracy: ", accuracy)