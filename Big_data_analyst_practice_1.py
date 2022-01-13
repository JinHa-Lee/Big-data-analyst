# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




df = pd.read_csv('./mtcars.csv',index_col=[0])
df_group = df.groupby(['cyl'])
print(df_group)
max_hp_df = df_group['hp'].mean().max()
print(max_hp_df)

df_subset = df[['cyl','mpg','hp','drat','qsec']]
max_skew_df = df_subset.apply(skew).max()
print(max_skew_df)

df_qsec = df[['qsec']]
minmax = MinMaxScaler()
minmax.fit(df_qsec)
minmax_df = minmax.transform(df_qsec)
print(minmax_df)
print(sum(minmax_df>0.5)[0])


print(df_subset.corr(method='pearson').apply(abs))

df_disp = df[['disp']]
log_df = np.log(df_disp+1)
print(log_df.max()-log_df.min())
