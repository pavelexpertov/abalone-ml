# In [ ]
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# In [ ]
df = pd.read_csv('abalone.data', header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings'])

# In []
X = df.loc[:, ['Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']].to_numpy()
y = np.ravel(df.loc[:, ['Sex']].to_numpy())
enc_y = LabelEncoder()
enc_y.fit(np.ravel(df.loc[:, ['Sex']].to_numpy()))
encoded_y = enc_y.transform(np.ravel(df.loc[:, ['Sex']].to_numpy()))

# In [ ]
SCORES_RESULTS = []
def add_score(name, mean_val, st_dev):
    SCORES_RESULTS.append([name, "mean: {}".format(mean_val), "std: {}".format(st_dev)])

# In [ ]
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
add_score("DecisionTreeClassifier", scores.mean(), scores.std())

# In [ ]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=280)
scores = cross_val_score(clf, X, y, cv=10)
add_score("LogisticRegression", scores.mean(), scores.std())

# In [ ]
from sklearn.svm import SVC
clf = SVC()
scores = cross_val_score(clf, X, y, cv=10)
add_score("SVC", scores.mean(), scores.std())

# In [ ]
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
scores = cross_val_score(clf, X, y, cv=10)
add_score("SGDClassifier", scores.mean(), scores.std())

# In [ ]
from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=150000)
scores = cross_val_score(clf, X, y, cv=10)
add_score("LinearSVC", scores.mean(), scores.std())

# In [ ]
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
scores = cross_val_score(clf, X, y, cv=10)
add_score("RandomForestClassifier", scores.mean(), scores.std())

# In [ ]
from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
# Need to debug this.
# scores = cross_val_score(clf, X, encoded_y, cv=10)
#add_score("CategoricalNB", scores.mean(), scores.std())

# In [ ]
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
scores = cross_val_score(clf, X, y, cv=10)
add_score("NearestCentroid", scores.mean(), scores.std())

# In [ ]
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
scores = cross_val_score(clf, X, y, cv=10)
add_score("KNeighborsClassifier", scores.mean(), scores.std())

# In [ ]
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter=700)
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With MLPClassifier', 'Mean:', scores.mean(), scores.std())
scores.mean()
scores.std()
recording_list.append(['With MLPClassifier', scores.mean(), scores.std()])

# In [ ]
for n, m, s in sorted(SCORES_RESULTS, key=lambda x: x[1]):
    print(n, m, s)
