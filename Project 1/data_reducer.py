import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn import over_sampling, under_sampling
from imblearn.pipeline import Pipeline

train_transaction_data_file = "./dataset/train_transaction.csv"
test_transaction_data_file = "./dataset/test_transaction.csv"
train_identity_data_file = "./dataset/train_identity.csv"
test_identity_data_file = "./dataset/test_identity.csv"
sample_submission_file = "./dataset/sample_submission.csv"

train_transaction_data = pd.read_csv(train_transaction_data_file)
train_identity_data = pd.read_csv(train_identity_data_file)
test_transaction_data = pd.read_csv(test_transaction_data_file)
test_identity_data = pd.read_csv(test_identity_data_file)
sample_submission = pd.read_csv(sample_submission_file)
del train_transaction_data_file, test_transaction_data_file, train_identity_data_file, test_identity_data_file, sample_submission_file


# reduce memory data function
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


train_transaction_data = reduce_mem_usage(train_transaction_data)
train_identity_data = reduce_mem_usage(train_identity_data)
test_transaction_data = reduce_mem_usage(test_transaction_data)
test_identity_data = reduce_mem_usage(test_identity_data)


train_transaction_data.to_pickle('train_transaction_data.pkl')
train_identity_data.to_pickle('train_identity_data.pkl')
test_transaction_data.to_pickle('test_transaction_data.pkl')
test_identity_data.to_pickle('test_identity_data.pkl')
