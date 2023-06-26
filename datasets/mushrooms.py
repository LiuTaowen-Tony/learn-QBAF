from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def load_mushroom(is_fuzzy: bool):
    data = pd.read_csv("./datasets/mushrooms.csv", header=0, na_values="?")
    # drop column with missing values
    data.drop("stalk_root", axis=1, inplace=True)
    data.head()
    X_, y = data.iloc[:, 1:23], data.iloc[:, 0]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = pd.get_dummies(X_)

    inputs = X.columns.values
    label = ["edible", "poisonous"]
    X = X.to_numpy()

    return X, y, inputs, label
