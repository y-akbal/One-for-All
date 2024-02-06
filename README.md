# One for All: A Method for forecasting multiple time series in a univariate sense using a single model.
Some files related to the upcoming manuscript which revolves around ensemble learning for time series forecasting using transformers.
1) layers.py contains some of the layers to be used. For the sake of completeness we implement our own layers,
2) preprocess_train_test.py will contain necessary functions to grab the data out of csv files and do some preprocessing and prepare them,
3) convert_memmap.py wll convert will grab csv files convert them into data.dat and lengths.dat files.
4) memmap_arrays.py is a signature snippet, grabbing data.dat and lengths.dat files, and will give you a window of given size from your concatted time series, as X[:-1], X[-1] and place_, where place_ indicates which time series to be used.
5) RF.py is used to do old-school machine learning analysis -  given that all csv.files are preprocessed - you can choose for instance randomforest or KNN sort of old school stuff. Splitting of time series is done on the fly. Multiprocessing is supported. At the end of the procedure, a file will be provided to see the results.
6) validate.py is for validation purposses, for instance to see R^2, MSE, MAE, MAPE values on both train and validation set.

  



<p align="center">

<img src="assets/memes.png" width="512" class="center"/>

</p>
