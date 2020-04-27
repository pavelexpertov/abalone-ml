# In [ ]
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# In [ ]
df = pd.read_csv('abalone.data', header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings'])

df['Mcubed_per_abalone_weight'] = (df['Height'] * df['Diameter'] * df['Length']) / df['Whole']
df['Mcubed_per_shucked_weight'] = (df['Height'] * df['Diameter'] * df['Length']) / df['Shucked']
df['Mcubed_per_viscera_weight'] = (df['Height'] * df['Diameter'] * df['Length']) / df['Viscera']
df['Mcubed_per_shell_weight'] = (df['Height'] * df['Diameter'] * df['Length']) / df['Shell']

# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=280)
scores = cross_val_score(clf, X, y, cv=10)
print("LogisticRegression Mean:", scores.mean(), "STD:" scores.std())
