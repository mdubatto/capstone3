import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer


data = pd.read_csv('../data/train.csv')
feats = pd.read_csv('../data/features.csv')
stores = pd.read_csv('../data/stores.csv')

data.Date = pd.to_datetime(data.Date)
feats.Date = pd.to_datetime(feats.Date)

data = data.drop(columns='IsHoliday').merge(feats, how='left', on=['Store','Date'])
data = data.merge(stores, how='left', on=['Store'])
data['Month'] = data['Date'].dt.month
data.drop(columns=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], inplace=True)

mask = data['Date'] <= '2011-10-30'
train = data[mask]
test = data[~mask]

cont_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment','Size']
disc_cols = ['Store', 'Dept', 'Type', 'Month']

transformers = [([cont_col], StandardScaler()) for cont_col in cont_cols]
encode_lst = [([disc_col], OneHotEncoder(sparse=False)) for disc_col in disc_cols]
transformers.extend(encode_lst)

mapper = DataFrameMapper(transformers, default=None, df_out=True)
train_scale = mapper.fit_transform(train)
test_scale = mapper.transform(test)

train_scale.to_csv('../data/train_clean.csv')
test_scale.to_csv('../data/test_clean.csv')