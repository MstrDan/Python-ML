# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:29:36 2020

@author: damar
"""

import numpy as np
import pandas as pd
import pprint 
import matplotlib.pyplot as plt
import re





#for year in Years:
#datasetpivot = dataset.stack()
        
#Load data if no header
#header = ['Country','Year','Value', 'Value Footnotes']
#Populations = pd.read_csv(Path+r'\UNData_Population.csv', sep=',', header=1, names=header,engine='python')

# No file path load
#Titanic_Train = pd.read_csv('train.csv', sep=',',engine='python')


# file path load
Path=r"C:\Users\damar\Documents\Python Scripts\Titanic"
Titanic_Train = pd.read_csv(Path+r'\train.csv', sep=',',engine='python')
Titanic_Test = pd.read_csv(Path+r'\test.csv', sep=',',engine='python')
Titanic_Train.head(20)
Titanic_Test.head()
Titanic_Train.info()
Titanic_Train['Sex'].head()
#Reorder class to 1st class better than 3
Mapper = {1:3, 3:1}
Titanic_Train['Pclass'] = Titanic_Train['Pclass'].replace(Mapper)
Median = Titanic_Train['Age'].median()

#Is Married woman indicator
Titanic_Train['Is_Married_Fem'] = [ 1 if 'Mrs' in x else 0 for x in Titanic_Train['Name']]
#Is First Class
Titanic_Train['Is_Firstclass'] = [ 1 if x == 1 else 0 for x in Titanic_Train['Pclass']]

#Convert <1 ages
Titanic_Train['Is_kid'] = [ 1 if 'Master' in x else 0 for x in Titanic_Train['Name']]

#Has >3 siblings spouse
Titanic_Train['Has_Siblings'] = [ 1 if x > 2 else 0 for x in Titanic_Train['SibSp']]

#Has >3 Has Parent children
Titanic_Train['Has_Parent'] = [ 1 if x > 1 else 0 for x in Titanic_Train['Parch']]

#Is Elderly
#Titanic_Train['Is_Elderly'] = [ 1 if x > 60 else 0 for x in Titanic_Train['Age']]

#Is High Fare > 100
#Titanic_Train['Is_Highfare'] = [ 1 if x > 100 else 0 for x in Titanic_Train['Fare']]

#S_embarked
#Titanic_Train['S_Embarked'] = [ 1 if x not in ['C','Q'] else 0 for x in Titanic_Train['Embarked']]

Titanic_Train.head()
Titanic_Train.info()

Mapper = {np.nan:9}
#Titanic_Train['Age'] = Titanic_Train[Titanic_Train['Is_kid'] == 1]['Age'].replace(Mapper)
#Titanic_Train['Age']=Titanic_Train[Titanic_Train['Age'] < 1]['Age'].map(lambda x:3)

#Titanic_Train['Age'] = Titanic_Train['Age'].map(lambda x:3)
Titanic_Train[Titanic_Train['Is_kid']==1] = Titanic_Train[Titanic_Train['Is_kid']==1].replace(np.nan, 9)

#Titanic_Train[Titanic_Train['PassengerId'] == 79]['Age'].head()

Titanic_Train['Age'] = [ 3 if x<1 else x for x in Titanic_Train['Age']]

#X = Titanic_Train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Is_Married_Fem', 'Is_Firstclass', 'Is_kid']]
#X = Titanic_Train.iloc[:, [2,4,5,6,7,9,11,12,13,14,15,16]].values
X = Titanic_Train.iloc[:, [2, 4,5,9,11,12,13,14,15,16]].values

y = Titanic_Train.iloc[:,1].values

#print(X[:,6])
#print (y)

#X[X['Age'].isnull()]
#Impute Null age using Mean strategy
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:,2:3])

#Impute Null fare using Mean strategy
imputer.fit(X[:,3:4])
X[:,3:4] = imputer.transform(X[:,3:4])

#Impute Null embark using most_frequent strategy
imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
imputer.fit(X[:,4:5])
X[:,4:5] = imputer.transform(X[:,4:5])

#pd.get_dummies(Titanic_Train['Sex'])

#Encode Sex Label
from sklearn.preprocessing import LabelEncoder
le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])


#Dummy variables
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [4])
#X = onehotencoder.fit_transform(X).toarray()


#print(X)
#Onehotencoder Gender 
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
#X = np.array(ct.fit_transform(X))



#Onehotencoder Embark field S, C, Q
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[4])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
#Remove one dummy variable
X = X[:,1:]
#print(X[0:6])




#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=0)

#Apply scaling using Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X =  sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
#Apply scaling using Normalization
from sklearn.preprocessing import Normalizer
sc = Normalizer()
#X =  sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""


#Apply Artificial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def create_network():

    #Initialize the ANN
    classifier = Sequential()

    #Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim = 11))

    #Adding the second hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation = 'relu'))

    #Adding the output layer
    classifier.add(Dense(output_dim=1, init='uniform', activation = 'sigmoid'))

    #Compiling the ANN
    classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
    #Root mean square propagation
    #classifier.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    #Fitting the ANN to the training set
    #classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

    return classifier

neural_network = KerasClassifier(build_fn=create_network, epochs=10, batch_size=5, verbose=0)

#Create k-fold cross-validation
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=10, shuffle=True, random_state =1)
score = cross_val_score(neural_network, X_train, y_train, cv=kf).mean()
print(score)


"""
#Predict test set result
y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
y_pred = [ 1 if x > 0.5 else 0 for x in y_pred]


#Making confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
"""

