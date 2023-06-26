import sklearn.datasets

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import binning, fuzzy_binning, create_csv_with_header
import pandas as pd

def load_iris(is_fuzzy: bool):
    # Loading iris data
    iris_data = sklearn.datasets.load_iris()
    x = iris_data.data
    y_ = iris_data.target.reshape(-1, 1)  # convert data to a single column

    # one hot encode class labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y_)

    # binning of features
    X, inputs = binning(
        pd.DataFrame(x),
        n_bins=3,
        encode="onehot-dense",
        strategy="uniform",
        feature_names=["Petal length", "Petal width", "Sepal length", "Sepal width"],
    )

    X_fuzzy, inputs_fuzzy = fuzzy_binning(
        pd.DataFrame(x),
        n_bins=3,
        feature_names=["Petal length", "Petal width", "Sepal length", "Sepal width"],
    )

    if is_fuzzy:
        X = X_fuzzy
        inputs = inputs_fuzzy
    label = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    X = X.to_numpy()
    y = y.astype(float)
    return X, y, inputs, label

_, _, _, label = load_iris(is_fuzzy=True)

print(label)