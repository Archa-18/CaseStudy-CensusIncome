import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('adult.csv')

lr = LogisticRegression(random_state=0)
rf = RandomForestClassifier(random_state=1)
dt = DecisionTreeClassifier(random_state=0)
sv = SVC()
nb = MultinomialNB()
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=0)
gb = GradientBoostingClassifier(n_estimators=10)

data.drop('native.country', inplace=True, axis=1)

le = LabelEncoder()
data['workclass'] = le.fit_transform(data['workclass'])
data['education'] = le.fit_transform(data['education'])
data['race'] = le.fit_transform(data['race'])
data['occupation'] = le.fit_transform(data['occupation'])
data['marital.status'] = le.fit_transform(data['marital.status'])
data['relationship'] = le.fit_transform(data['relationship'])
data['sex'] = le.fit_transform(data['sex'])
data['income'] = le.fit_transform(data['income'])

data = data.replace('?', np.nan)

x = data.drop(['race', 'relationship', 'marital.status', 'income'], axis=1)
y = data['income']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

lr.fit(x_train, y_train)
rf.fit(x_train, y_train)
dt.fit(x_train, y_train)
sv.fit(x_train, y_train)
nn.fit(x_train, y_train)
gb.fit(x_train, y_train)
nb.fit(x_train, y_train)

lr_predict = lr.predict(x_test)
rf_predict = rf.predict(x_test)
dt_predict = dt.predict(x_test)
sv_predict = sv.predict(x_test)
nn_predict = nn.predict(x_test)
gb_predict = gb.predict(x_test)
nb_predict = nb.predict(x_test)

print('Logistic', accuracy_score(y_test, lr_predict))
print('RandomForest', accuracy_score(y_test, rf_predict))
print('DecisionTree', accuracy_score(y_test, dt_predict))
print('SVM', accuracy_score(y_test, sv_predict))
print('NeuralNetwork', accuracy_score(y_test, nn_predict))
print('GradientBoostingClassifier', accuracy_score(y_test,  gb_predict))
print('NaiveBayes', accuracy_score(y_test,  nb_predict))

#Logistic 0.7980345992425018
#RandomForest 0.8356024158050978
#DecisionTree 0.7860579383764971
#SVM 0.7949636605589109
#NeuralNetwork 0.24147814515303512
#GradientBoostingClassifier 0.8120585525642338
#NaiveBayes 0.7833964581840516