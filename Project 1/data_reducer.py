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


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']
#https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest_transaction_data-579654
for c in ['P_emaildomain', 'R_emaildomain']:
    train_transaction_data[c + '_bin'] = train_transaction_data[c].map(emails)
    test_transaction_data[c + '_bin'] = test_transaction_data[c].map(emails)
    
    train_transaction_data[c + '_suffix'] = train_transaction_data[c].map(lambda x: str(x).split('.')[-1])
    test_transaction_data[c + '_suffix'] = test_transaction_data[c].map(lambda x: str(x).split('.')[-1])
    
    train_transaction_data[c + '_suffix'] = train_transaction_data[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test_transaction_data[c + '_suffix'] = test_transaction_data[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')






numerical = [col for col in numerical if col in train_transaction_data.columns]
categorical = [col for col in categorical if col in train_transaction_data.columns]


def nan2mean(df):
    for x in list(df.columns.values):
        if x in numerical:
            #print("___________________"+x)
            #print(df[x].isna().sum())
            df[x] = df[x].fillna(0)
           #print("Mean-"+str(df[x].mean()))
    return df
train_transaction_data=nan2mean(train_transaction_data)
test_transaction_data=nan2mean(test_transaction_data)


category_counts = {}
for f in categorical:
    train_transaction_data[f] = train_transaction_data[f].replace("nan", "other")
    train_transaction_data[f] = train_transaction_data[f].replace(np.nan, "other")
    test_transaction_data[f] = test_transaction_data[f].replace("nan", "other")
    test_transaction_data[f] = test_transaction_data[f].replace(np.nan, "other")
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_transaction_data[f].values) + list(test_transaction_data[f].values))
    train_transaction_data[f] = lbl.transform(list(train_transaction_data[f].values))
    test_transaction_data[f] = lbl.transform(list(test_transaction_data[f].values))
    category_counts[f] = len(list(lbl.classes_)) + 1
# train_transaction_data = train_transaction_data.reset_index()
# test_transaction_data = test_transaction_data.reset_index()



for column in numerical:
    scaler = StandardScaler()
    if train_transaction_data[column].max() > 100 and train_transaction_data[column].min() >= 0:
        train_transaction_data[column] = np.log1p(train_transaction_data[column])
        test_transaction_data[column] = np.log1p(test_transaction_data[column])
    scaler.fit(np.concatenate([train_transaction_data[column].values.reshape(-1,1), test_transaction_data[column].values.reshape(-1,1)]))
    train_transaction_data[column] = scaler.transform(train_transaction_data[column].values.reshape(-1,1))
    test_transaction_data[column] = scaler.transform(test_transaction_data[column].values.reshape(-1,1))



train_transaction_data.to_pickle('train_transaction_data.pkl')
train_identity_data.to_pickle('train_identity_data.pkl')
test_transaction_data.to_pickle('test_transaction_data.pkl')
test_identity_data.to_pickle('test_identity_data.pkl')
