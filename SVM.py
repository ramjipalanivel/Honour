import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  


# Read dataset to pandas dataframe
irisdata = pd.read_csv('iris.csv')  

X = irisdata.drop('Species', axis=1)  
y = irisdata['Species']  


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 0)  

from sklearn.svm import SVC  
Classifier = SVC(kernel='linear',random_state = 0)  
Classifier.fit(X_train, y_train)  


y_pred = Classifier.predict(X_test)  

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

#Polynomial Kernerl

from sklearn.svm import SVC  
svSpeciesifier = SVC(kernel='poly', degree=8)  
svSpeciesifier.fit(X_train, y_train) 
y_pred = svSpeciesifier.predict(X_test)  

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')



#Sigmoid Kernel

from sklearn.svm import SVC  
svSpeciesifier = SVC(kernel='sigmoid')  
svSpeciesifier.fit(X_train, y_train)  

y_pred = svSpeciesifier.predict(X_test)  

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

#Gaussian Kernel

from sklearn.svm import SVC  
svSpeciesifier = SVC(kernel='rbf')  
svSpeciesifier.fit(X_train, y_train)  

y_pred = svSpeciesifier.predict(X_test)  

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')