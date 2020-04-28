# In [ ]
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# In [ ]
df = pd.read_csv('abalone.data', header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings'])

df['Mcubed_per_whole_weight'] = (df['Height'] * df['Diameter'] * df['Length']) / df['Whole']
df['Mcubed_per_shucked_weight'] = (df['Height'] * df['Diameter'] * df['Length']) / df['Shucked']
df['Mcubed_per_viscera_weight'] = (df['Height'] * df['Diameter'] * df['Length']) / df['Viscera']
df['Mcubed_per_shell_weight'] = (df['Height'] * df['Diameter'] * df['Length']) / df['Shell']

# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=320)
scores = cross_val_score(
    clf,
    df.loc[:, ['Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']].to_numpy(),
    np.ravel(df.loc[:, ['Sex']].to_numpy()),
    cv=10
)
print("LogisticRegression Mean:", scores.mean(), "STD:", scores.std())

# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=320)
scores = cross_val_score(
    clf,
    df.loc[:, ['Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell']].to_numpy(),
    np.ravel(df.loc[:, ['Sex']].to_numpy()),
    cv=10
)
print("LogisticRegression Mean:", scores.mean(), "STD:", scores.std())

# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=320)
scores = cross_val_score(
    clf,
    df.loc[:, ['Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings', 'Mcubed_per_shell_weight', 'Mcubed_per_shucked_weight', 'Mcubed_per_viscera_weight', 'Mcubed_per_whole_weight']].to_numpy(),
    np.ravel(df.loc[:, ['Sex']].to_numpy()),
    cv=10
)
print("LogisticRegression Mean:", scores.mean(), "STD:", scores.std())


# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=320)
scores = cross_val_score(
    clf,
    df.loc[:, ['Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Mcubed_per_shell_weight', 'Mcubed_per_shucked_weight', 'Mcubed_per_viscera_weight', 'Mcubed_per_whole_weight']].to_numpy(),
    np.ravel(df.loc[:, ['Sex']].to_numpy()),
    cv=10
)
print("LogisticRegression Mean:", scores.mean(), "STD:", scores.std())


# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=320)
scores = cross_val_score(
    clf,
    df.loc[:, ['Length', 'Diameter', 'Height']].to_numpy(),
    np.ravel(df.loc[:, ['Sex']].to_numpy()),
    cv=10
)
print("LogisticRegression Mean:", scores.mean(), "STD:", scores.std())


# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=320)
scores = cross_val_score(
    clf,
    df.loc[:, ['Length', 'Diameter', 'Height', 'Mcubed_per_shell_weight', 'Mcubed_per_shucked_weight', 'Mcubed_per_viscera_weight', 'Mcubed_per_whole_weight']].to_numpy(),
    np.ravel(df.loc[:, ['Sex']].to_numpy()),
    cv=10
)
print("LogisticRegression Mean:", scores.mean(), "STD:", scores.std())


# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=320)
scores = cross_val_score(
    clf,
    df.loc[:, ['Whole', 'Shucked', 'Viscera', 'Shell']].to_numpy(),
    np.ravel(df.loc[:, ['Sex']].to_numpy()),
    cv=10
)
print("LogisticRegression Mean:", scores.mean(), "STD:", scores.std())


# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=320)
scores = cross_val_score(
    clf,
    df.loc[:, ['Whole', 'Shucked', 'Viscera', 'Shell', 'Mcubed_per_shell_weight', 'Mcubed_per_shucked_weight', 'Mcubed_per_viscera_weight', 'Mcubed_per_whole_weight']].to_numpy(),
    np.ravel(df.loc[:, ['Sex']].to_numpy()),
    cv=10
)
print("LogisticRegression Mean:", scores.mean(), "STD:", scores.std())

# <markdown>
Notes:
- Original Features
    * The original features produced 55% percent accuracy and std of 5.3% (With and without 'Ring' attribute didn't affect at all since regress was 1%).
    * With engineered features, the accuracy was 56% with std of 6.4%. Slight improvement. With or without 'Ring' attribute didn't affect performance ac    curacy whatsover.
- The dimension measurements:
    * On their own, it was 51.4% accruacy with 0.02% std.
    * with engineered features of mm/gram rates, accuracy was 53% with 0.02 std. Slight improvement.
- The weight measurements:
    * On their own, it was 54% accuracy with std of 4.5% std.
    * with engineered features of mm/gram rates, accuracy was 54% with 4.5% std.  Pretty much no improvement.

Deductions:
    - It looks like the top accuracy is with original features. Engineered features improved it by 1%.
    - Dimension and weight measurements couldn't deliver top accuracy compared to combined: differences is 3.6% and 1% respectively.
    - What's interesting that dimension measurements were improved with weight-dimension-rates features to give it extra 2% accuracy. I guess that weights with dimensions help to determine gender of the shell better.
