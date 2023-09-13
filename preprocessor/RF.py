import concurrent.futures
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from preprocess import get_csv_list, preprocess_csv
import os
import pandas as pd
import pickle
import time


def return_split(ts: np.ndarray, split_ratio: float = 0.8, lags: int = 16):
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
    regressor=LinearRegression,
    **kwargs,
):
    preprocessed_ts, _ = preprocess_csv(csv_file, numerical_column=numerical_column)
    numerical_ts = preprocessed_ts.iloc[:, numerical_column].to_numpy()

    X_train, y_train, X_test, y_test = return_split(
        numerical_ts, split_ratio=split_ratio, lags=lags
    )
    np.random.seed(seed)
    rf = regressor(**kwargs)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print(f"{csv_file, score, len(numerical_ts)} Done")
    return score, csv_file


def main():
    csv_list = get_csv_list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        res = executor.map(return_final_result, csv_list)
    return res


def short(len_=2048):
    list_ = get_csv_list()
    names = []
    lengths = []
    for csv_fil in list_:
        pd_ser = pd.read_csv(csv_fil, encoding="ISO-8859-1", decimal=",")
        len__ = len(pd_ser)
        if len__ < len_:
            names.append(csv_fil)
            lengths.append(len__)
    return names, lengths


def return_average(file="pickled.t"):
    with open(file, "rb") as file_:
        L = pickle.load(file_)
    temp_av = 1e-2
    k = 0
    list_outlier = []
    for M in L:
        value, name = M
        if 0 < value < 1:
            temp_av += value
            k += 1
        else:
            list_outlier.append(name)
    return temp_av / k, list_outlier


if __name__ == "__main__":
    a = time.time()
    res = main()
    list_ = list(res)
    with open("pickled.t", mode="wb") as file:
        pickle.dump(list_, file)
    av = return_average()
    print(av)
