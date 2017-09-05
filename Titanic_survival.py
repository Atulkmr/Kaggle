import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
   

titanic_df = pd.read_csv('input/train.csv')
test_df    = pd.read_csv('input/test.csv')

#  print(titanic_df.info())
#  titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].max())


