# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import seaborn as sns
import os

df = pd.read_csv('./mpg.csv')

mpg_to_kpl = 1.60934/3.78541

df['kpl'] = df['mpg']*mpg_to_kpl
df['kpl'] = df['kpl'].round(2)

df.info()
df['horsepower'].unique()

df['horsepower'].replace('?',np.nan,inplace=True)
df.dropna(subset=['horsepower'],axis=0,inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

df.info()


df['origin'].replace({1:'USA',2:'EU',3:'JPN'},inplace=True)
count, bin_dividers = np.histogram(df['horsepower'],bins=3)
print(bin_dividers)

bin_names = ['저출력','보통출력','고출력']


df['hp_bin'] = pd.cut(x = df['horsepower'],
                        bins = bin_dividers,
                        labels = bin_names,
                        include_lowest=True)

horsepower_dummies = pd.get_dummies(df['hp_bin'])
print(horsepower_dummies)
