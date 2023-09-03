import concurrent.futures
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from preprocess import get_csv_list, preprocess_csv
import os
import pandas as pd


def return_split(ts: np.ndarray, split_ratio: float = 0.8, lags: int = 512):
    """_summary_

    Args:
        ts (np.ndarray): _description_
        split_ratio (float, optional): _description_. Defaults to 0.8.
        lags (int, optional): _description_. Defaults to 512.

    Returns:
        _type_: _description_
    """
    N = len(ts)
    N_ = int(split_ratio * N)
    ts_train = ts[:N_]
    ts_test = ts[N_:]
    N_train = len(ts_train)
    N_test = len(ts_test)
    ## Do some malloc thing
    X_train = np.zeros((N_train - lags, lags))
    y_train = np.zeros(N_train - lags)
    ##
    X_test = np.zeros((N_test - lags, lags))
    y_test = np.zeros(N_test - lags)
    ##
    for i in range(N_train - lags):
        X_train[i] = ts_train[i : i + lags]
        y_train[i] = ts_train[i + lags]
    for i in range(N_test - lags):
        X_test[i] = ts_test[i : i + lags]
        y_test[i] = ts_test[i + lags]
    return X_train, y_train, X_test, y_test


def return_final_result(
    csv_file,
    split_ratio=0.8,
    lags: int = 256,
    numerical_column=-1,
    seed=0,
    regressor=RandomForestRegressor,
    **kwargs,
):
    preprocessed_ts, _ = preprocess_csv(csv_file, numerical_column=numerical_column)
    numerical_ts = preprocessed_ts.iloc[:, numerical_column].to_numpy()[:5000]
    X_train, y_train, X_test, y_test = return_split(
        numerical_ts, split_ratio=split_ratio, lags=lags
    )
    np.random.seed(seed)
    rf = regressor(**kwargs)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print(f"{csv_file} Done")
    return score, csv_file


def main():
    csv_list = get_csv_list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        res = executor.map(return_final_result, csv_list)
    return res


if __name__ == "__main__":
    res = main()
    print(list(res))
