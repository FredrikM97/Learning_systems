import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from copy import deepcopy
from itertools import product
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve,validation_curve

def get_data_info(input_train, target_train):
    # The shape should look like (401,133) where 401 is the number of features and 133 is each row
    print("Length of",len(input_train))
    print("X.shape:", input_train.shape, "y.shape:", target_train.shape)    
    print("Contains Nan:",np.isnan(input_train).any(), np.isnan(target_train).any())
    print("Contains +inf:",np.isinf(input_train).any(),np.isinf(target_train).any())
    print("Contains -inf:",np.isneginf(input_train).any(),np.isneginf(target_train).any())
    #pd.DataFrame(input_train).describe()
    
def preprocessing(input_train, target_train, input_test):
    input_train_copy = deepcopy(input_train)

    # In case one-hot'
    if len(target_train.T) == 3:

        print("One hot --> single value output")
        rows, cols = np.where(target_train == 1)
        target_train = cols


    print(len(target_train.T))
    # Normalizing data
    scaler = StandardScaler()
    scaler.fit(input_train,y=target_train)

    input_train_copy_normalized = deepcopy(input_train_copy)

    input_train = scaler.transform(input_train)
    input_test = scaler.transform(input_test)
   
    return input_train,input_test,target_train,input_train_copy,input_train_copy_normalized

def feature_reduction(input_train, target_train,input_train_copy):
    pca = PCA(n_components = .95, svd_solver = 'full')
    pca.fit(input_train,y=target_train)

    pca_input = pca.transform(input_train)

    feature_tot = len(pca_input[0])
    print("original shape:   ", input_train_copy.shape)
    print("transformed shape:", pca_input.shape)
    print("Explained variance:",pca.explained_variance_ratio_)
    
    return feature_tot, pca, pca_input

def plot_feature_variance(pca_input):
    cntItems = 1
    rowItems = 1
    fig, ax = plt.subplots(cntItems,rowItems,figsize=(5,5))

    if cntItems == 1: axlist = range(rowItems)
    else: axlist = list(product(range(cntItems),range(rowItems)))

    # Plot feature variance
    plt.title("Feature variance")
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    feature_variance = np.var(pca_input, 0)
    plt.plot(feature_variance)

    plt.show()

def plot_top_features(feature_tot,pca_input):
    startpos = 1
    cntPlots = length if (length := feature_tot) <= 9 else 9
    rowItems = 3
    cntItems = ceil(cntPlots/rowItems)

    fig, ax = plt.subplots(cntItems,rowItems,figsize=(15,10))
    if cntItems == 0: axlist = range(rowItems)
    else: axlist = list(product(range(cntItems),range(rowItems)))


    for index in range(0,(cntPlots)):
        ax[axlist[index]].set_title("PCA - Histogram - Feature: " + str(index))
        ax[axlist[index]].hist(pca_input[:,index])

    plt.show()
    
def feature_selection(score_function, input_train, target_train, input_test, feature_tot='all'):
    # Using amount of features based on PCA information
    fs = SelectKBest(score_func=score_function, k=feature_tot)
    fs.fit(input_train, target_train)
    input_train_fs = fs.transform(input_train)
    input_test_fs = fs.transform(input_test)

    print(input_train.shape)
    plt.title("Feature selection")
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.show()
    
    return input_train_fs, input_test_fs

def parameter_tuning(estimators, param_grid, scoring, input_train_fs, target_train, k=10):
    pipeline = Pipeline(estimators)
    grid = GridSearchCV(
        pipeline,
        cv=k, 
        param_grid=param_grid,
        return_train_score=True,
        refit=True,
        n_jobs=-1,
        scoring=scoring
    ) 
    grid.fit(input_train_fs,target_train)
    return grid

def get_model_info(grid):
    param_cols = ['']
    score_cols = ['mean_train_score', 'std_test_score','mean_test_score', 'std_test_score']

    grid_df = pd.DataFrame(grid.cv_results_, columns=score_cols)
    grid_df.sort_values(by=['mean_test_score']).tail()
    #print(grid_df)

    print(f"Best score: {grid.best_score_}\nBest params {grid.best_params_}")
    
def validate_curve(grid, input_train_fs, target_train,scoring=None, k=10):
    # Parameter to
    param_range = np.arange(0.01, 50,5)
    model = grid.best_params_['clf']
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    plot_learning_curve(model, model.__class__.__name__, input_train_fs, target_train, axes=axes[:], cv=k, scoring=scoring, n_jobs=-1)
    plt.show()
    
def predict_model(grid, input_train_fs, target_train, input_test_fs):
    model = grid.best_params_['clf']
    model.fit(input_train_fs, target_train)
    print(model.predict(input_test_fs))
    return model

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, scoring=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, 
                                                                               X, 
                                                                               y, 
                                                                               cv=cv, 
                                                                               n_jobs=n_jobs, 
                                                                               scoring=scoring,
                                                                               train_sizes=train_sizes,
                                                                               return_times=True
                                                                              )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
