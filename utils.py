import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# to use ADSTuner
from ads.hpo.search_cv import ADSTuner
from ads.hpo.stopping_criterion import *
from ads.hpo.distributions import *

# to encode categoricals
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import make_scorer

#
# supporting functions
#
def add_features(df):
    # feature engineering
    df_new = df.copy()
    
    df_new['datetime'] = pd.to_datetime(df_new['datetime'])
    
    # this way I add 2 engineered features
    df_new['hour'] = df_new['datetime'].dt.hour
    df_new['year'] = df_new['datetime'].dt.year
    
    return df_new

#
# define a custom scorer for ADSTuner
# aligned with the scorer used in Kaggle leaderboard
#
def rmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

#
# functions for categorical encoding
#

# first train label encoder
TO_CODE = ['season', 'weather', 'year']

def train_encoders(df):
    le_list = []

    for col in TO_CODE:
        print(f'train for coding: {col} ')
    
        le = LabelEncoder()
        le.fit(df[col])
        
        le_list.append(le)
    
    print()
    
    return le_list

# then use it
def apply_encoders(df, le_list):

    for i, col in enumerate(TO_CODE):
        print(f'Coding: {col} ')
    
        le = le_list[i]
    
        df[col] = le.transform(df[col])
    
    # special treatment for windspeed
    # windpeed actually is integer badly rounded !!
    # print('Coding: windspeed')
    # df['windspeed'] = np.round(df['windspeed'].values).astype(int)
    
    return df

def show_tuner_results(tuner):
    
    # to count completed
    result_df = tuner.trials[tuner.trials['state'] == 'COMPLETE'].sort_values(by=['value'], ascending=False)
    
    print("ADSTuner session results:")
    print(f"ADSTuner has completed {result_df.shape[0]} trials")
    print()
    print(f"The best trial is the #: {tuner.best_index}")
    print(f"Parameters for the best trial are: {tuner.best_params}")
    print(f"The metric used to optimize is: {tuner.scoring_name}")
    print(f"The best score is: {round(tuner.best_score, 4)}")
    
def show_categoricals(df, thr):
    THR = thr
    FIGSIZE = (9, 6)
    # to get cols in alfabetical order
    cols = sorted(df.columns)

    # changed using list comprehension, to shorten code
    cols2 = [col for col in cols if df[col].nunique() < THR]
    list_count2 = [
        df[col].nunique() for col in cols if df[col].nunique() < THR
    ]

    # plot
    plt.figure(figsize=FIGSIZE)
    plt.title("Low cardinality features")
    ax = sns.barplot(x=cols2, y=list_count2)
    # to plot values on bar
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=90)
    plt.ylabel("# of distinct values")
    plt.grid(True)
 