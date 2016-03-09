# References:
# http://pandas.pydata.org/pandas-docs/stable/missing_data.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/titanic.csv')
# df = pd.read_csv('./data/titanic-only10.csv')

# With "info" we get a insight of how are our data,
# for example, how many data we have in each variable.
# For example we can see that there aren't too many data in
# room and boat variable, and we have just half of ages.
df.info()
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 1313 entries, 0 to 1312
# Data columns (total 11 columns):
# row.names    1313 non-null int64
# pclass       1313 non-null object
# survived     1313 non-null int64
# name         1313 non-null object
# age          633 non-null float64
# embarked     821 non-null object
# home.dest    754 non-null object
# room         77 non-null object
# ticket       69 non-null object
# boat         347 non-null object
# sex          1313 non-null object

# Stage 1) Cleaning the data
#


# A lot of Machine learning algorithms are going to need numeric
# values, so we converted string values to int
df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
# df['embarked'] = df['embarked'].fillna('Unknown')
# df['embarked'] = df['embarked'].map({'Cherbourg': 1, 'Queenstown': 2, 'Southampton': 3, 'Unknown': 4}).astype(int)
df['age'] = df['age'].fillna(df['age'].mean())
df['pclass'] = df['pclass'].dropna().map({'1st': 1, '2nd': 2, '3rd': 3}).astype(int)


# We could do a deeper analysis and figure out where we have more data with NaN age.
# If we do that, it come out that a better value for this NaN should be the age mean
# of male passenger of third class.
# 1) First we figure out how many NaN we have taking car of class and sex
# df[np.isnan(df['age']) == True].groupby(['pclass', 'sex'])['survived'].count()
# pclass  sex
# 1st     0       42
#         1       54
# 2nd     0       22
#         1       46
# 3rd     0      156
#         1      360
# 2) We figure out the mean for each group and for the whole dataset, in order
#    to check it if there are relevant differences between them
# df[np.isnan(df['age']) == False].groupby(['pclass', 'sex'])['age'].mean()
# pclass  sex
# 1st     0      37.772277
#         1      41.199334
# 2nd     0      27.388235
#         1      28.910761
# 3rd     0      22.564328
#         1      25.327294
# df[np.isnan(df['age']) == False]['age'].mean()
# 31.19418104265403


# If age variable were of String type, we could do something like
# m = df['age'].dropna().astype(float).mean()
# df['age'] = df['age'].fillna(m).astype(float)


# Stage 2) Analyzing the data
#

# A interesting data could have been the room number, but we don't have enough information.

_ = sns.pairplot(df, vars=['pclass', 'sex'], hue="survived", palette='Set1')
_ = sns.lmplot('sex', 'pclass', df, x_jitter=.15, y_jitter=.15, hue="survived", palette="Set1", fit_reg=False)
# sns.pairplot(df, vars=['embarked', 'sex'], hue="survived", palette='Set1')
# sns.lmplot('sex', 'embarked', df, x_jitter=.15, y_jitter=.15, hue="survived", palette="Set1", fit_reg=False);
plt.figure(figsize=(12, 10))
_ = sns.corrplot(df, annot=False)



