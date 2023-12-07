import numpy as np
import pandas as pd

import os, random, time
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer

from LSTCN_1 import LSTCN

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)
    random.seed(42)
def load_data(df, n_steps, split=0.8):
    """ Prepare the time series for learning.

    Parameters
    ----------
    df      :   {dataframe} path to the CSV (variables must appear by column)
    n_steps :   {int} Number of steps-ahead to be forecast.
    split   :   {float} Proportion of data used for training.
    Returns
    ----------
    X_train, Y_train, X_test, Y_test, n_features

    """

    for col in df.columns:
        df[col] = 0.01 + 0.98 * (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    n_features = len(df.columns)
    data = df.to_numpy()

    # splitting the data to create the datasets
    data_train = data[:(int(split * data.shape[0])) + n_steps, :]
    data_test = data[-int((1 - split) * data.shape[0]):, :]

    # creating X_train, Y_train, X_test, Y_test
    X_train, Y_train = create_dataset(data_train, n_features, n_steps)
    X_test, Y_test = create_dataset(data_test, n_features, n_steps)

    return data_train, X_train, Y_train, X_test, Y_test, n_features
def create_dataset(data, n_features, n_steps):
    """ Create X and Y from a portion of the time series.

    Parameters
    ----------
    data         :   {array-like} Portion of the time series.
    n_features   :   {int} Number of features in the time series. 
    n_steps      :   {int} Number of steps-ahead to be forecast.
    Returns
    ----------
    X_train, Y_train or X_test, Y_test

    """

    X_data = []
    Y_data = []

    for index in range(0, data.shape[0] - (2 * n_steps - 1)):
        # the moving windows is set to one
        X_data.append(data[index:(index + n_steps), :])
        Y_data.append(data[(index + n_steps):(index + 2 * n_steps), :])

    # reshape data into 2D-NumPy arrays
    X = np.reshape(np.array(X_data), (len(X_data), n_steps * n_features))
    Y = np.reshape(np.array(Y_data), (len(Y_data), n_steps * n_features))

    return X, Y



df = pd.read_csv('data' + os.sep + 'ETTh1.csv')
df.drop(labels=df.columns[0], axis='columns', inplace=True)
# df = df.diff().dropna()

n_steps = 1
data, X_train, Y_train, X_test, Y_test, n_features = load_data(df, n_steps)

"""Initialize initial state of the model (Random)"""
#The initial state of the model is set to random values between -0.5 and 0.5
#Using pre-trained weights can (possibly) improve the performance of the model
np.random.seed(42)
W1 = -0.5 + np.random.rand(
    n_steps * n_features + 1, n_steps * n_features)


param_search = {
    'alpha': [1.0E-3, 1.0E-2, 1.0E-1],
    'n_blocks': range(2, 6)
}
def one_step_scorer(Y_true, Y_pred):
        
    Y_true_ = Y_true[:, n_features-1::n_features]
    Y_pred_ = Y_pred[:, n_features-1::n_features]
    
    return mean_absolute_error(Y_true_, Y_pred_)

reset_random_seeds()
tscv = TimeSeriesSplit(n_splits=5)
model = LSTCN(n_features, n_steps, W1 = W1)

start = time.time()
scorer = make_scorer(model.score, greater_is_better=False)
gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, refit=True,
                       n_jobs=-1, error_score='raise', scoring=scorer)
gsearch.fit(X_train, Y_train)
best_model = gsearch.best_estimator_
end = time.time()

Y_pred = best_model.predict(X_test)

train_error = round(one_step_scorer(best_model.predict(X_train), Y_train), 4)
test_error = round(one_step_scorer(Y_pred, Y_test), 4)

print(str(train_error) + ',' + str(test_error) + ',' + str(round(end - start, 4)))

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '16'

print(Y_test.shape)
fig, ax = plt.subplots(figsize=(18, 6))

plt.plot(Y_test[:, -1], label='ground-truth', color='tab:brown')
plt.plot(Y_pred[:, -1], label='predicted', color='tab:blue')

leg = ax.legend(loc='upper right')

plt.savefig('example_pred.png', dpi=300, bbox_inches='tight')
plt.show()

""""Feature importance"""
W2 = best_model.stcn.W2
def feature_importance(W2, n_steps, n_features):
    """ Compute the feature importance matrix.

    Parameters
    ----------
    W2          :   {array-like} of shape (n_features*n_steps, n_features*n_steps)
                    The weight matrix of the LSTCN model.
    n_steps     :   {int} Number of steps-ahead to be forecast.
    n_features  :   {int} Number of features in the time series. 
    Returns
    ----------
    feature_importance : {array-like} of shape (n_features, n_steps)
                         The feature importance matrix.

    """
    #cut bias
    W2 = W2[:-1, :]
    
    feature_importance = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            for k in range(n_steps):
                for l in range(n_steps):
                    feature_importance[j, i] = feature_importance[j, i] + np.abs(W2[i + k * n_features, j + l * n_features])

    # normalize the feature importance matrix
    for i in range(n_features):
        feature_importance[:, i] = np.round(feature_importance[:, i] / np.sum(feature_importance[:, i]),2)
    
    return feature_importance

feature_importance = feature_importance(W2, n_steps, n_features)

def feature_importance_matrix(feature_importance):
    """ Visualize the feature importance matrix.

    Parameters
    ----------
    feature_importance : {array-like} of shape (n_features, n_steps)
                         The feature importance matrix.

    """
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(feature_importance)

    # We want to show all ticks...
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    # ... and label them with the respective list entries
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(n_features):
        for j in range(n_features):
            text = ax.text(j, i, feature_importance[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Feature Importance Matrix")
    fig.tight_layout()
    plt.show()

feature_importance_matrix(feature_importance)