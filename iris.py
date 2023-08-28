import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df = pd.read_excel('Iris.xls')

print(df.head())




#Data Splitting

x = df.iloc[:,0:4]
y = df.iloc[:,4]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)




#Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_cm = confusion_matrix(y_test, lr_pred)

#According to the confusion matrix, our logistic regression model got 49 of the 50 data correctly.




#K-NN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_cm = confusion_matrix(y_test, knn_pred)

#According to the confusion matrix, our KNN classifier model got 49 of the 50 data correctly.




#SVM

from sklearn import svm as SVM

svm = SVM.SVC(kernel='linear')
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)      
svm_cm = confusion_matrix(y_test, svm_pred)

#According to the confusion matrix, our SVC with linear kernel model got all of the 50 data correctly.




#Naive Bayes

from sklearn.naive_bayes import CategoricalNB

gnb = CategoricalNB()
gnb.fit(x_train, y_train)
gnb_pred = gnb.predict(x_test)
gnb_cm = confusion_matrix(y_test, gnb_pred)

#According to the confusion matrix, our Categorical Naive Bayes model got 49 of the 50 data correctly.




#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)
dt_cm = confusion_matrix(y_test, dt_pred)

#According to the confusion matrix, our decision tree model got 48 of the 50 data correctly.




#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, criterion='entropy')
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_cm = confusion_matrix(y_test, rf_pred)

#According to the confusion matrix, our random forest model got 48 of the 50 data corectly.









