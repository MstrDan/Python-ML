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


search_type = "video"


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
Titanic_Test.head()
#Reorder class to 1st class better than 3
#Mapper = {1:3, 3:1}
#Titanic_Train['Pclass'] = Titanic_Train['Pclass'].replace(Mapper)
#Titanic_Test['Pclass'] = Titanic_Test['Pclass'].replace(Mapper)
#Is Married woman indicator
Titanic_Train['Is_Married_Fem'] = [ 1 if 'Mrs' in x else 0 for x in Titanic_Train['Name']]
Titanic_Test['Is_Married_Fem'] = [ 1 if 'Mrs' in x else 0 for x in Titanic_Test['Name']]

#Is First Class
Titanic_Train['Is_Firstclass'] = [ 1 if x == 1 else 0 for x in Titanic_Train['Pclass']]
Titanic_Test['Is_Firstclass'] = [ 1 if x == 1 else 0 for x in Titanic_Test['Pclass']]

#Convert <1 ages
Titanic_Train['Is_kid'] = [ 1 if 'Master' in x else 0 for x in Titanic_Train['Name']]
Titanic_Test['Is_kid'] = [ 1 if 'Master' in x else 0 for x in Titanic_Test['Name']]

#Has >3 siblings spouse
Titanic_Train['Has_Siblings'] = [ 1 if x > 1 else 0 for x in Titanic_Train['SibSp']]
Titanic_Test['Has_Siblings'] = [ 1 if x > 1 else 0 for x in Titanic_Test['SibSp']]

#Has >3 Has Parent children
Titanic_Train['Has_Parent'] = [ 1 if x > 1 else 0 for x in Titanic_Train['Parch']]
Titanic_Test['Has_Parent'] = [ 1 if x > 1 else 0 for x in Titanic_Test['Parch']]

Titanic_Train['Has_Family'] = Titanic_Train['SibSp'] + Titanic_Train['Parch'] 
Titanic_Test['Has_Family'] = Titanic_Test['SibSp'] + Titanic_Test['Parch'] 


#Titanic_Train['Is_Elderly'] = [ 1 if x > 60 else 0 for x in Titanic_Train['Age']]
#Titanic_Test['Is_Elderly'] = [ 1 if x > 60 else 0 for x in Titanic_Test['Age']]

Mapper = {np.nan:5}
#Titanic_Train['Age'] = Titanic_Train[Titanic_Train['Is_kid'] == 1]['Age'].replace(Mapper)
#Titanic_Test['Age'] = Titanic_Test[Titanic_Test['Is_kid'] == 1]['Age'].replace(Mapper)

#Titanic_Train['Age'] = [ 5 if x<1 else x for x in Titanic_Train['Age']]
#Titanic_Test['Age'] = [ 5 if x<1 else x for x in Titanic_Test['Age']]

#Titanic_Train[Titanic_Train['Is_kid']==1] = Titanic_Train[Titanic_Train['Is_kid']==1].replace(np.nan, 9)
#Titanic_Test[Titanic_Test['Is_kid']==1] = Titanic_Test[Titanic_Test['Is_kid']==1].replace(np.nan, 9)

#Titanic_Train[Titanic_Train['PassengerId'] == 79]['Age'].head()

#Titanic_Train['Age'] = [ 3 if x<1 else x for x in Titanic_Train['Age']]
#Titanic_Test['Age'] = [ 3 if x<1 else x for x in Titanic_Test['Age']]
Titanic_Train.info()

#X = Titanic_Train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Is_Married_Fem', 'Is_Firstclass', 'Is_kid']]
#X_train = Titanic_Train.iloc[:, [2,4,5,9,11,12,13,14,15,16]].values
#y_train = Titanic_Train.iloc[:,1].values

#X_train = Titanic_Train.loc[:, ['Pclass', 'Sex','Age','Fare','Embarked','Is_Married_Fem','Is_kid','Has_Siblings','Has_Parent']]
X_train = Titanic_Train.loc[:, ['Pclass','Name', 'Sex','Age','Fare','Embarked','Is_Married_Fem', 'Is_kid', 'Has_Family']]
y_train = Titanic_Train.loc[:,['Survived']]

#X_test = Titanic_Test.loc[:, ['Pclass', 'Sex','Age','Fare','Embarked','Is_Married_Fem','Is_kid','Has_Siblings','Has_Parent']]
X_test = Titanic_Test.loc[:, ['Pclass','Name', 'Sex','Age','Fare','Embarked','Is_Married_Fem', 'Is_kid', 'Has_Family']]
#X_test = Titanic_Test.iloc[:, [1,3,4,8,10,11,12,13, 14,15]].values
#print(X_test)
#y_test = Titanic_Test.iloc[:,1].values





median_list = list(X_train.groupby('Pclass')['Age'].median().values)
for i in range(3):
    X_train.loc[X_train['Pclass']==i+1,'Age'] = X_train.loc[X_train['Pclass']==i+1,'Age'].fillna(median_list[i])

#median_list = list(X_test.groupby('Pclass')['Age'].median().values)
for i in range(3):
    X_test.loc[X_test['Pclass']==i+1,'Age'] = X_test.loc[X_test['Pclass']==i+1,'Age'].fillna(median_list[i])


X_train['age_range'] = pd.cut(X_train['Age'], bins=[0,2,17,55,99], labels=['Baby', 'Child', 'Adult', 'Elderly'])
X_test['age_range'] = pd.cut(X_test['Age'], bins=[0,2,17,55,99], labels=['Baby', 'Child', 'Adult', 'Elderly'])


#Impute Null age using Mean strategy
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
#imputer.fit(X_train.loc[:,['Age']])
#X_train.loc[:,['Age']] = imputer.transform(X_train.loc[:,['Age']])
#X_test.loc[:,['Age']] = imputer.transform(X_test.loc[:,['Age']])

#Impute Null fare using Mean strategy
imputer.fit(X_train.loc[:,['Fare']])
X_train.loc[:,['Fare']] = imputer.transform(X_train.loc[:,['Fare']])
X_test.loc[:,['Fare']] = imputer.transform(X_test.loc[:,['Fare']])



#Impute Null embark using most_frequent strategy
imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
imputer.fit(X_train.loc[:,['Embarked']])
X_train.loc[:,['Embarked']] = imputer.transform(X_train.loc[:,['Embarked']])
X_test.loc[:,['Embarked']] = imputer.transform(X_test.loc[:,['Embarked']])



#Encode Sex Label
from sklearn.preprocessing import LabelEncoder
le_1 = LabelEncoder()
X_train.loc[:, ['Sex']] = le_1.fit_transform(X_train.loc[:, ['Sex']])
X_test.loc[:, ['Sex']] = le_1.transform(X_test.loc[:, ['Sex']])



title_mapping = {'Don':'Rare','Rev':'Rare','Mme':'Miss','Ms':'Miss',
                 'Major':'Rare','Lady':'Royal','Mlle':'Miss','Col':'Rare','Capt':'Rare',
                 'Sir':'Royal', 'Countess':'Royal', 'Jonkheer':'Royal', 'Dona':'Royal'}

for i in X_train['Name']:
    X_train['Title'] = X_train['Name'].str.extract('([A-Za-z]+)\.',expand=True)
# Dropping Name
X_train.drop(columns=['Name'],inplace=True)
# Replacing by mapping
X_train.groupby('Title').count()

X_train.replace({'Title':title_mapping},inplace=True)

for i in X_test['Name']:
    X_test['Title'] = X_test['Name'].str.extract('([A-Za-z]+)\.',expand=True)
# Dropping Name
X_test.drop(columns=['Name'],inplace=True)
# Replacing by mapping
#X_test.groupby('Title').count()
X_test.replace({'Title':title_mapping},inplace=True)



#print(X_test.Title.unique())

#Onehotencoder Pclass field 1, 2, 3
dummies = pd.get_dummies(X_train['Pclass'], prefix='Pclass', drop_first= True)
X_train= pd.concat([X_train, dummies], axis=1)
X_train = X_train.drop('Pclass', axis=1)
dummies = pd.get_dummies(X_test['Pclass'], prefix='Pclass', drop_first= True)
X_test= pd.concat([X_test, dummies], axis=1)
X_test = X_test.drop('Pclass', axis=1)

#Onehotencoder Title Mr, Mrs, Miss, Master, Rare, Dr
dummies = pd.get_dummies(X_train['Title'], prefix='Title', drop_first= True)
X_train= pd.concat([X_train, dummies], axis=1)
X_train = X_train.drop('Title', axis=1)
dummies = pd.get_dummies(X_test['Title'], prefix='Title', drop_first= True)
X_test= pd.concat([X_test, dummies], axis=1)
X_test = X_test.drop('Title', axis=1)

dummies = pd.get_dummies(X_train['age_range'], prefix='age_range', drop_first= True)
X_train= pd.concat([X_train, dummies], axis=1)
X_train = X_train.drop('age_range', axis=1)
dummies = pd.get_dummies(X_test['age_range'], prefix='age_range', drop_first= True)
X_test= pd.concat([X_test, dummies], axis=1)
X_test = X_test.drop('age_range', axis=1)

#Onehotencoder Embark field S, C, Q
dummies = pd.get_dummies(X_train['Embarked'], prefix='Embarked', drop_first= True)
X_train= pd.concat([X_train, dummies], axis=1)
#Removed column already encoded
X_train = X_train.drop('Embarked', axis=1)

dummies = pd.get_dummies(X_test['Embarked'], prefix='Embarked', drop_first= True)
X_test= pd.concat([X_test, dummies], axis=1)
#Removed column already encoded
X_test = X_test.drop('Embarked', axis=1)


#Train test split
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

#Apply scaling using Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X =  sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




import xgboost as xgb
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test)

"""
param = {
    'eta': 0.3, 
    'max_depth': 15,  
    'gamma': 0.4,
    'min_child_weight': 5,
    'objective': 'multi:softmax',  
    'num_class': 2} 
"""
param = {
    'eta': 0.15, 
    'max_depth': 10,  
    'gamma': 0.2,
    'min_child_weight': 5,
    "objective": 'multi:softmax',  
    'num_class': 2,
    'n_jobs':0,
    'random_state':0,
    "colsample_bytree":0.5
     } 
steps = 30  # The number of training iterations
#model = xgb.train(param, D_train, steps)
"""
model = xgb.XGBClassifier(objective='multi:softmax', eta=0.15, 
                          gamma=0.2,random_state=0, n_estimators=100, max_depth=15,
                          min_child_weight=5, max_features=0.7, 
                          num_class =2,
                          min_samples_leaf=0.6, importance_type='gain',
                          learning_rate=0.5, colsample_bytree=0.6)
"""
model = xgb.XGBClassifier(objective='reg:logistic', eta=0.05, 
                          gamma=0.4,random_state=0, n_estimators=100, max_depth=5,
                          min_child_weight=1, max_features=0.4, 
                          base_score = 0.5,
                          min_samples_leaf=0.6, importance_type='gain',
                          learning_rate=0.1, colsample_bytree=0.3)



model.fit(X_train, y_train)

#Predict test set result
y_pred = model.predict(X_test)
#y_pred = np.asarray([np.argmax(line) for line in D_pred])



header = ['Survived']

Titanic_Pred = pd.concat([Titanic_Test,pd.DataFrame(y_pred, columns=header)],axis=1)

Titanic_csvfile = Path+'\Titanic_submission41.csv'

Titanic_Pred[['PassengerId', 'Survived']].to_csv(Titanic_csvfile, mode='w', index=False)    

