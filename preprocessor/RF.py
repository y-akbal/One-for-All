import concurrent.futures
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from preprocess_train_test import get_csv_list, preprocess_csv
import pandas as pd
from matplotlib import pyplot as plt


def return_split(ts: np.ndarray, split_ratio: float = 0.8, lags: int = 4):
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
    lags: int = 16,
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
    return csv_file, score, numerical_ts

def create_save_fig(scores:list):
    plt.hist(scores, density = True, bins = 50, label = "R^2 values")
    plt.legend()
    plt.savefig('my_plot.png')

def main(score_fie = "score.csv"):
    csv_list = get_csv_list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        res = executor.map(return_final_result, csv_list)
    
    scores = {}
    score_vals = []
    for csv_file, score, len_ in res:
        scores[csv_file] = score
        score_vals.append(float(score))
    
    data_frame = pd.DataFrame().from_dict(scores, orient = "index")
    data_frame.to_csv("results_scores.csv")
    ## -- ##
    create_save_fig(score_vals)
    return None

if __name__ == "__main__":
    main()
