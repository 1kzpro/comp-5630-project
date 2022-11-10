from sklearn.model_selection import train_test_split
import pandas as pd

def generate_data(filepath, sep=";"):
    raw_df = pd.read_csv(filepath, sep=sep)

    for column in raw_df.columns:
        column_dtype = raw_df[column].dtype
        if str(column_dtype) == "object":
            # print(f"Column: {column} and type {column_dtype}")
            dummy = pd.get_dummies(raw_df[column], prefix=column, drop_first=True)
            raw_df = pd.concat([raw_df, dummy], axis=1).drop(column, axis=1)

    print("Pre-Processed Training Dataset", raw_df)

    X = raw_df.iloc[:,:-1]
    y = raw_df.iloc[:,-1]

    return train_test_split(X, y, test_size=0.33, random_state=42)