from base64 import encode
import warnings

import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")


def group_2(l : list):
    """
    Groups a list into pairs of two elements
    """
    return list(zip(l, l[1:]))

def fuzzy_binning(features, n_bins: int, feature_names):
    """
    fuzzy binning algorithm as described in report
    using the triangular membership function
    assign fuzzy values to each fuzzy set
    n_bins: number of bins
    feature_names: list of feature names
    """
    X = []
    binning_feature_names = []
    for i in range(features.shape[1]):
        x = features.iloc[:, i]
        # size : num_bins + 2
        quantiles = np.quantile(x, np.linspace(0, 1, n_bins + 2))
        # size : len(x) * num_bins
        y = np.zeros((len(x), n_bins), dtype=np.float32)

        intervals = group_2(quantiles) 
        middle_bin_ranges = list(zip(quantiles[1:-3], quantiles[3:]))


        y[x < quantiles[1], 0] = 1

        # between quantiles i to (i+1) where i in [1..n_bins-1]
        # quantile i to (i+1) corresponding to bin (i-1) and bin i
        for (q_left, q_right), bin in zip(intervals[1:-1], range(1, n_bins)):
            selector = (q_left <= x) & (x < q_right)
            selected_x = x[selector]
            interested_vale = (selected_x - q_left) / (q_right - q_left)
            y[selector, bin] = interested_vale
            y[selector, bin - 1] = 1 - interested_vale

        y[quantiles[-2] <= x, -1] = 1

        X.append(y)
        binning_feature_names.append(f"{feature_names[i]} bin 0, x < {quantiles[2]}")
        for enum, (q_left, q_right) in enumerate(middle_bin_ranges):
            binning_feature_names.append(
                f"{feature_names[i]} bin {enum + 1}, {q_left} <= x < {q_right}"
            )
        binning_feature_names.append(
            f"{feature_names[i]} bin {n_bins - 1}, x >= {quantiles[-3]}"
        )
    df = pd.DataFrame(np.concatenate(X, axis=1))
    return df, binning_feature_names

def binning(features, n_bins, strategy, encode, feature_names):
    """
    Returns binned features and the corresponding labels for each bin
    'n_bins' can either be an integer or a list/numpy array of n integers (different number of bins for n features)
    'strategy' and 'encode' are inputs for Scikit-learns KBinsDiscretizer
    """
    X = []
    binning_feature_names = []
    for i in range(features.shape[1]):
        x = features.iloc[:, i]
        if isinstance(n_bins, (list, np.ndarray)):
            num_bins = n_bins[i]
        elif isinstance(n_bins, int):
            num_bins = n_bins
        else:
            raise ValueError("`n_bins` should be a an integer, list or numpy array.")
        est = KBinsDiscretizer(n_bins=num_bins, encode=encode, strategy=strategy)
        x = est.fit_transform(x.values.reshape(-1, 1))

        bin_edges = est.bin_edges_[0]
        X.append(x)
        for j in range(num_bins):
            if j == 0:
                fname = feature_names[i] + " x < " + "{:.1f}".format(bin_edges[j + 1])
            elif j == num_bins - 1:
                fname = (
                    feature_names[i] + " " + "{:.1f}".format(bin_edges[j]) + "$\leq x$"
                )
            else:
                fname = (
                    feature_names[i]
                    + " "
                    + "{:.1f}".format(bin_edges[j])
                    + "$\leq x <$"
                    + "{:.1f}".format(bin_edges[j + 1])
                )
            binning_feature_names.append(fname)
    df = pd.DataFrame(np.concatenate(X, axis=1))
    return df, binning_feature_names


def create_csv_with_header(fname):
    """Creates a csv to store the results and writes a header."""
    with open(fname, "w") as file:
        writer = csv.writer(file)
        header = [
            "Parameters",
            "Number of connections",
            "Training accuracy",
            "Test accuracy",
            "Recall",
            "Precision",
            "F1 score",
        ]
        writer.writerow(header)
