import pandas as pd
import numpy as np
diabetes = pd.read_csv('diabetes.csv')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], 
diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test,y_test)))
