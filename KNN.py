from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import binarize
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# features info : https://goo.gl/p8ocBn
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# initialization of the Dataset
data = pd.read_csv('pima-indians-diabetes.data', names=col_names)
# checking if there is not empty fields (SKL require none empty field)
data.count()
data.head()
# selection of relevant features
label = data['label']
data.drop('label', axis=1, inplace=True)

X, y = data, label
# distributing our Dataset into a training and testing distribution
# we use the default SKL split (0.75 (75%) for training)
# random_state=0 Setting the random seed (for reproductibility purpose)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# counting the split of Positive (1) and Negative (0) labels in our testing distribution
y_test.value_counts()