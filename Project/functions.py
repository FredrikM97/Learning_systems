import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from math import ceil
from copy import deepcopy
from itertools import product
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve,validation_curve
from sklearn.model_selection import train_test_split 
from joblib import dump, load
import sys
from warnings import filterwarnings
from pandas.plotting import scatter_matrix

"""Global"""
plt.rcParams['figure.figsize'] = [15, 15]
pd.set_option('display.max_colwidth',None)
plt.rcParams.update({'font.size': 14})
np.set_printoptions(threshold=sys.maxsize)
filterwarnings('ignore')

def get_data_info(input_train, target_train):
    """
    Prints the shape and warn if input contains Nan, +inf or -inf
    Expected output contains a tuple of (data points, features)
    
    Parameters
    ----------
    input_train: 2D-list 
    target_train: 1D-list
    
    """
    
    print("Length of",len(input_train))
    print("X.shape:", input_train.shape, "y.shape:", target_train.shape)    
    print("Contains Nan:",np.isnan(input_train).any(), np.isnan(target_train).any())
    print("Contains +inf:",np.isinf(input_train).any(),np.isinf(target_train).any())
    print("Contains -inf:",np.isneginf(input_train).any(),np.isneginf(target_train).any())
    #print(f"Input: {input_train[:2]} \nTarget: {target_train[:2]}")
    
    #print(pd.DataFrame(input_train).describe()-9
    
def preprocessing(input_train, target_train, input_test):
    """
    Normalize the data
    
    Parameters
    ----------
    input_train: 2D-list 
    target_train: 1D-list
    test_train: 1D-list
    """
    input_train_copy = deepcopy(input_train)

    # In case one-hot'
    if len(target_train.T) == 3:

        print("One hot --> single value output")
        rows, cols = np.where(target_train == 1)
        target_train = cols


    print("Length of input:", len(target_train.T))
    # Normalizing data
    scaler = StandardScaler()
    scaler.fit(input_train,y=target_train)

    input_train_copy_normalized = deepcopy(input_train_copy)

    input_train = scaler.transform(input_train)
    input_test = scaler.transform(input_test)
   
    return input_train,input_test,target_train,input_train_copy,input_train_copy_normalized

def feature_reduction(input_train, target_train,input_train_copy):
    """
    Evaluate the input data with PCA transforms. returns the new shape and number of features
    
    Parameters
    ----------
    input_train: 2D-list
    target_train: 2D-list
    input_train_copy: 2D-list
    
    Return
    ----------
    feature_tot: int
        Number of features
    pca: Object
        Contains the Pca fit base in input_train and target_train
        
    pca_input: 2D-list
        The transformed input_train
    """
    
    pca = PCA(n_components = .95, svd_solver = 'full')
    pca.fit(input_train,y=target_train)

    pca_input = pca.transform(input_train)

    feature_tot = len(pca_input[0])
    print("original shape:   ", input_train_copy.shape)
    print("transformed shape:", pca_input.shape)
    print("Explained variance:",pca.explained_variance_ratio_)
    
    return feature_tot, pca, pca_input

def plot_feature_variance(pca_input, filedir, taskname):
    """
    Plot the feature variance 
    
    Parameters
    ----------
    pca_input: 2D-list
    """
    cntItems = 1
    rowItems = 1
    fig, ax = plt.subplots(cntItems,rowItems,figsize=(5,5))

    if cntItems == 1: axlist = range(rowItems)
    else: axlist = list(product(range(cntItems),range(rowItems)))

    # Plot feature variance
    plt.title(taskname + " - Feature variance")
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    feature_variance = np.var(pca_input, 0)
    plt.plot(feature_variance)
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_f_variance.png", format='png')
    
    plt.show()

def plot_top_features(feature_tot,pca_input, filedir, taskname,k=9):
    """
    Plot the top features up to k features
    
    Parameters
    ----------
    pca_input: 2D-list
    k: int, Optional, default: 9
        Number of wanted feature plots
    """
    startpos = 1
    cntPlots = length if (length := feature_tot) <= 9 else k
    rowItems = 3
    cntItems = ceil(cntPlots/rowItems)

    fig, ax = plt.subplots(cntItems,rowItems,figsize=(15,10))
    
    if cntItems == 0: axlist = range(rowItems)
    else: axlist = list(product(range(cntItems),range(rowItems)))


    for index in range(0,(cntPlots)):
        ax[axlist[index]].set_title(taskname, "PCA - Histogram - Feature: " + str(index))
        ax[axlist[index]].set_xlabel('Frequency')
        ax[axlist[index]].set_ylabel('Value')
        ax[axlist[index]].hist(pca_input[:,index])
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_f_top.png", format='png')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def feature_selection(score_function, input_train, target_train, input_test, filedir, taskname, feature_tot='all'):
    """
    Select the best features (SelectKBest) based on score_function and dataset
    
    Parameters
    ----------
    score_function: str
        Score function for SelectKBest
    input_train: 2D-list
    target_train: list
    input_test: 2D-list
    feature_tot: int, Optional, default='all'
        Number of interesting features
    
    Return
    ----------
    input_train_fs: 2D-list
        Reduced 2D-list with number of wanted features
    input_test_fs: 2D-list
        Reduced 2D-list with number of wanted features
    """
    
    # Using amount of features based on PCA information
    fs = SelectKBest(score_func=score_function, k=feature_tot)
    fs.fit(input_train, target_train)
    cols = fs.get_support(indices=True)
    input_train_fs = fs.transform(input_train)
    input_test_fs = fs.transform(input_test)
    print(cols[:feature_tot])
    
    print("Shape of input:",input_train.shape)
    plt.title(taskname + " - Feature selection")
    plt.xlabel("Feature")
    plt.ylabel("Frequency")
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_f_select.png", format='png')
    plt.show()
    
    return input_train_fs, input_test_fs

def plot_feature_distribution(input_train_fs, filedir=None, taskname=None, datapoints=None):
    """
    Plot the distribution of features in boxplot
    
    Parameters
    ----------
    input_train_fs: 2D-list
        feature vector
    filedir: str, Optional, default=None
        Directory for saving file
    taskname: str, Optional, default=None
        Name of file to save plot
    datapoints: int, Optional, default=None
        How many points that should be displayed
    """
    if not datapoints: datapoints= len(input_train_fs) 
    dataset = pd.DataFrame(input_train_fs[:datapoints])
    #figs = dataset.hist()
    dataset.boxplot(grid=False)
    plt.xlabel("Features")
    plt.ylabel("Data values")
    plt.suptitle(taskname + " - Feature distribution")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_f_distribution.png", format='png')
    
    plt.show()
    
def plot_feature_relationship(input_train_fs, filedir=None, taskname=None):
    """
    Plot the relationship and combination of features 
    
    Parameters
    ----------
    input_train_fs: 2D-list
        feature vector
    filedir: str, Optional, default=None
        Directory for saving file
    taskname: str, Optional, default=None
        Name of file to save plot
    """
    print("x-axis contain features and y is frequency of values")
    dataset = pd.DataFrame(input_train_fs)
    dataset.hist(grid=False)
    plt.suptitle(taskname + " - Feature relationship")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_f_relationship.png", format='png')
    plt.plot()
    ax = scatter_matrix(dataset)
    plt.suptitle(taskname + " - Feature relationship")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_f_relationship_detailed.png", format='png')
    plt.show()
    
def parameter_tuning(estimators, param_grid, input_train_fs, target_train,scoring='accuracy', k=10, verbose=1):
    """
    Select the best features (SelectKBest) based on score_function and dataset
    
    Parameters
    ----------
    estimators: dict
        Models to train onto
    param_grid: dict
        Parameters based on the models
    scoring: str, Optional, default='accuracy'
        What kind on scoring method, default
    input_train_fs: 2D-list
    target_train: 2D-list
    k, int, Optional, default=10
        Number of k for cross validation
    verbose,int, Optional, default=1
        Debugging info level
    
    Return
    ----------
    grid: dict
        Data of the GridSearchCV
    """
    pipeline = Pipeline(estimators)
    grid = GridSearchCV(
        pipeline,
        cv=k, 
        param_grid=param_grid,
        return_train_score=True,
        refit=True,
        n_jobs=-1,
        scoring=scoring,
        verbose=verbose
    ) 
    grid.fit(input_train_fs,target_train)
    return grid

def get_model_info(grid):
    """
    Print info of the grid models and their hyperparameters.
    
    Parameters
    ----------
    grid: Dict
    
    """
    param_cols = ['']
    score_cols = ['mean_train_score', 'std_test_score','mean_test_score', 'std_test_score']

    grid_df = pd.DataFrame(grid.cv_results_, columns=score_cols)
    grid_df.sort_values(by=['mean_test_score']).tail()
    #print(grid_df)

    #print(f"Best score: {grid.best_score_}\nBest params {grid.best_params_}\n")
    
    
    #means = grid.cv_results_['mean_test_score']
    #stds = grid.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    #    print("%0.3f (+/-%0.03f)\n"% (mean, std * 2))
    
    df = pd.DataFrame(list(grid.cv_results_['params']))
    ranking = grid.cv_results_['rank_test_score']
    # The sorting is done based on the test_score of the models.
    sorting = np.argsort(grid.cv_results_['rank_test_score'])

    # Sort the lines based on the ranking of the models
    df_final = df.iloc[sorting]
    frame = pd.DataFrame(grid.cv_results_).loc[:, ['mean_test_score', 'std_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score')
    print(frame.head())

def validate_curve(grid, input_train_fs, target_train, filedir, taskname, scoring=None, k=10):
    """
    Validate the best grid model based on training and crossvalidation.
    
    Parameters
    ----------
    grid: Object
        GridSearchCV object
    input_train_fs: 2D-list
    target_train: 2D-list
    scoring: str, Optional, default=None:
        Evaluation metric for scoring
    k: int,Optional,default=10
        Number of k for crossvalidation 
    
    """
    param_range = np.arange(0.01, 50,5)
    model = grid.best_params_['clf']
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    plot_learning_curve(model, taskname +" - "+model.__class__.__name__, input_train_fs, target_train, axes=axes[:], cv=k, scoring=scoring, n_jobs=-1)
    fig.tight_layout()
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_validation.png", format='png')
    
    plt.show()
    
def save_model(model, filedir=None, taskname=None):
    """
    Save model
    """
    if not model:
        print("Model is empty!!")
    else:
        file = filedir + "Models/" +  taskname + ".joblib"
        print("Save model into:",file)
        #with open(filedir + "Models/" +  taskname + ".joblib", 'wb') as file:
        dump(model, file) 
        
def load_model(filedir=None, taskname=None):
    """
    Load model
    """
    file = filedir + "Models/" +  taskname + ".joblib"
    print("Loading model: ",file)
    return load(file) 

def save_prediction(predict, filedir, taskname):
    """
    Save predictions
    """
    if not len(predict):
        print("Model is empty!!")
    else:
        file = filedir + "Predictions/" +  taskname + "_predict.txt"
        with open(file, "w") as output:
            output.write(str(predict))
    
def predict_model(grid, input_train_fs, target_train, input_test_fs):
    """
    Validate the best grid model based on training and crossvalidation.
    
    Parameters
    ----------
    grid: Object
        GridSearchCV object
    input_train_fs: 2D-list
    target_train: 2D-list
    input_test_fs: 2D-list
    """
    model = grid.best_params_['clf']
    model.fit(input_train_fs, target_train)
    predict = model.predict(input_test_fs)
    return model, predict

def display_model_predict(grid, input_train, target_train, filedir=None, taskname=None, datapoints=None):
    """
    Display prediction in scatter and line plot
    
    Parameters
    ----------
    grid: Object
        GridSearchCV object
    input_train_fs: 2D-list
    target_train: 2D-list
    filedir: str, Optional, default=None
        Directory for file
    taskname: str, Optional, default=None
        Name of file
    datapoints: str, Optional, default=None
        How many datapoints that should be displayed in plot
    """
    X_train, X_test, y_train, y_test = train_test_split(input_train, target_train, test_size=0.33, random_state=42)
    
    model, predict = predict_model(grid, X_train, y_train, X_test)
    
    if not datapoints: datapoints = len(predict)

    fig=plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.title(taskname +" - Predictions vs Expected value")
    plt.ylabel("Data Value")
    plt.xlabel('Feature')
    
    plt.scatter(range(datapoints),predict[:datapoints], marker='.', label="Predicted",color='red')
    plt.plot(y_test[:datapoints], label="Expected", color='blue')
    plt.legend()
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_prediction.png", format='png')
    plt.show()
    
def display_confusion_matrix(grid, input_train, target_train, filedir=None, taskname=None):
    """
    Display prediction in confusion matrix
    
    Parameters
    ----------
    grid: Object
        GridSearchCV object
    input_train_fs: 2D-list
    target_train: 2D-list
    filedir: str, Optional, default=None
        Directory for file
    taskname: str, Optional, default=None
        Name of file
    datapoints: str, Optional, default=350
        How many datapoints that should be displayed in plot
    """
    X_train, X_test, y_train, y_test = train_test_split(input_train, target_train, test_size=0.33, random_state=42)
    
    model, predict = predict_model(grid, X_train, y_train, X_test)
    
    cm = confusion_matrix(y_test,predict)
    
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
    ax.set_title("Confusion matrix - "  + taskname)
    if taskname and filedir: plt.savefig(filedir +"Pictures/"+ taskname + "_prediction.png", format='png')
    plt.show()

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, scoring=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        
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
    axes[0].set_xlabel("Data point")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Data point")
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
