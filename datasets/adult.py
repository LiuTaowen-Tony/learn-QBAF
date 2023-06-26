
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import binning, fuzzy_binning, create_csv_with_header


def load_adult(is_fuzzy: bool):
    filename = "./datasets/adult-all.csv"
    dataframe = read_csv(
        filename,
        header=None,
        na_values="?",
        names=[
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "Income",
        ],
    )
    # drop rows with missing
    dataframe = dataframe.dropna()
    target = dataframe.values[:, -1]
    # split into inputs and outputs
    last_ix = len(dataframe.columns) - 1
    X_, y = dataframe.drop("Income", axis=1), dataframe["Income"]
    # select categorical and numerical features
    cat_ix = X_.select_dtypes(include=["object", "bool"]).columns
    num_ix = X_.select_dtypes(include=["int64", "float64"]).columns
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    # one-hot encoding of categorical features
    df_cat = pd.get_dummies(X_[cat_ix])
    # binning of numerical features
    x = X_.drop(columns=cat_ix, axis=1)
    if is_fuzzy:
        df_num, num_list = fuzzy_binning(
            x,
            n_bins=3,
            feature_names=[
                "Age",
                "fnlwgt",
                "education-num",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
            ],
        )
    else:
        df_num, num_list = binning(
            x,
            n_bins=3,
            strategy="uniform",
            encode="onehot-dense",
            feature_names=[
                "Age",
                "fnlwgt",
                "education-num",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
            ],
        )
    X = pd.concat(
        [df_cat.reset_index(drop=True), pd.DataFrame(df_num).reset_index(drop=True)], axis=1
    )

    cat_label = df_cat.columns.values
    num_label = np.asarray(num_list)
    inputs = np.concatenate((cat_label, num_label), axis=0)
    label = ["Income $\leq$ 50K", "Income $>$ 50K"]

    X = X.astype(float)
    X = X.to_numpy()
    return X, y, inputs, label
