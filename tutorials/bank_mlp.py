from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

raw_df = pd.read_csv("data/bank/bank-full.csv", sep=";")

for column in raw_df.columns:
    column_dtype = raw_df[column].dtype
    if str(column_dtype) == "object":
        # print(f"Column: {column} and type {column_dtype}")
        dummy = pd.get_dummies(raw_df[column], prefix=column, drop_first=True)
        raw_df = pd.concat([raw_df, dummy], axis=1).drop(column, axis=1)

print("Pre-Processed Training Dataset", raw_df)

X = raw_df.iloc[:,:-1]
y = raw_df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(X_train_r)
# print(y_train_r)

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print("Accuracy: ", accuracy)