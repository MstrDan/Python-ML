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


import seaborn as sns



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
#Mapper = {1:3, 3:1}
#Titanic_Train['Pclass'] = Titanic_Train['Pclass'].replace(Mapper)
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

Titanic_Train['Has_Family'] = Titanic_Train['SibSp'] + Titanic_Train['Parch']
Titanic_Train['Is_Alone'] = [ 1 if x == 1 else 0 for x in Titanic_Train['Has_Family']]

#Titanic_Train[['Has_Family','Has_Siblings', 'Has_Parent']].head(10)
#Is Elderly
#Titanic_Train['Is_Elderly'] = [ 1 if x > 60 else 0 for x in Titanic_Train['Age']]

#Is High Fare > 100
#Titanic_Train['Is_Highfare'] = [ 1 if x > 100 else 0 for x in Titanic_Train['Fare']]

#S_embarked
#Titanic_Train['S_Embarked'] = [ 1 if x not in ['C','Q'] else 0 for x in Titanic_Train['Embarked']]

Titanic_Train.head()

Mapper = {np.nan:9}
#Titanic_Train['Age'] = Titanic_Train[Titanic_Train['Is_kid'] == 1]['Age'].replace(Mapper)
#Titanic_Train['Age']=Titanic_Train[Titanic_Train['Age'] < 1]['Age'].map(lambda x:3)

#Titanic_Train['Age'] = Titanic_Train['Age'].map(lambda x:3)
#Titanic_Train[Titanic_Train['Is_kid']==1] = Titanic_Train[Titanic_Train['Is_kid']==1].replace(np.nan, 9)

#Titanic_Train[Titanic_Train['PassengerId'] == 79]['Age'].head()

#Titanic_Train['Age'] = [ 3 if x<1 else x for x in Titanic_Train['Age']]

#X = Titanic_Train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Is_Married_Fem', 'Is_Firstclass', 'Is_kid']]
#X = Titanic_Train.iloc[:, [2,4,5,6,7,9,11,12,13,14,15,16]].values
#X = Titanic_Train.iloc[:, [2, 4,5,9,11,12,13,14,15,16]].values
#X = Titanic_Train.loc[:, ['Pclass', 'Sex','Age','Fare','Embarked','Is_Married_Fem','Is_Firstclass','Is_kid','Has_Siblings','Has_Parent']]
X = Titanic_Train.loc[:, ['Pclass','Name', 'Sex','Age','Fare','Embarked','Is_Married_Fem', 'Is_kid','Has_Family']]

X.head()

y = Titanic_Train.loc[:,['Survived']]




#print(X[:,6])
#print (y)

#X[X['Age'].isnull()]
#Impute Null age using Mean strategy
'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer.fit(X.loc[:,['Age']])
X.loc[:,['Age']] = imputer.transform(X.loc[:,['Age']])
'''

#print(X.groupby('Pclass')['Age'].median())
median_list = list(X.groupby('Pclass')['Age'].median().values)
#print("\nList of means of Ages grouped according to Pclass",median_list)

for i in range(3):
    X.loc[X['Pclass']==i+1,'Age'] = X.loc[X['Pclass']==i+1,'Age'].fillna(median_list[i])
print(X.Age.isnull().sum())

#td = td2
#Add Bins
X['age_range'] = pd.cut(X['Age'], bins=[0,2,17,55,99], labels=['Baby', 'Child', 'Adult', 'Elderly'])
#X['FareBin'] = pd.qcut(X['Fare'], 4,labels=['Low', 'Med', 'MedHigh', 'High'])

#
#Impute Null fare using Mean strategy
#imputer.fit(X[:,3:4])
#X[:,3:4] = imputer.transform(X[:,3:4])
from sklearn.impute import SimpleImputer
#Impute Null embark using most_frequent strategy
imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
imputer.fit(X.loc[:,['Embarked']])
X.loc[:,['Embarked']] = imputer.transform(X.loc[:,['Embarked']])

#pd.get_dummies(Titanic_Train['Sex'])

#Encode Sex Label
from sklearn.preprocessing import LabelEncoder
le_1 = LabelEncoder()
X.loc[:, ['Sex']] = le_1.fit_transform(X.loc[:, ['Sex']])


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
#tstd['Title'] = tstd['Name']

for i in X['Name']:
    X['Title'] = X['Name'].str.extract('([A-Za-z]+)\.',expand=True)
# Dropping Name
X.drop(columns=['Name'],inplace=True)
# Replacing by mapping
title_mapping = {'Don':'Rare','Rev':'Rare','Mme':'Miss','Ms':'Miss',
                 'Major':'Rare','Dona':'Royal','Mlle':'Miss','Col':'Rare','Capt':'Rare',
                 'Lady':'Royal', 'Sir':'Royal', 'Countess':'Royal', 'Jonkheer':'Royal'}



X.replace({'Title':title_mapping},inplace=True)
print(X.Title.unique())


'''
#Onehotencoder Embark field S, C, Q
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),['Embarked'])],remainder='passthrough')
#X = np.array(ct.fit_transform(X))
X = ct.fit_transform(X)

#Remove one dummy variable
#X = X[:,1:]
'''
dummies = pd.get_dummies(X['Pclass'], prefix='Pclass', drop_first= True)
X= pd.concat([X, dummies], axis=1)
X = X.drop('Pclass', axis=1)

dummies = pd.get_dummies(X['Title'], prefix='Title', drop_first= True)
X= pd.concat([X, dummies], axis=1)
X = X.drop('Title', axis=1)

dummies = pd.get_dummies(X['age_range'], prefix='age_range', drop_first= True)
X= pd.concat([X, dummies], axis=1)
X = X.drop('age_range', axis=1)

'''
dummies = pd.get_dummies(X['FareBin'], prefix='FareBin', drop_first= True)
X= pd.concat([X, dummies], axis=1)
X = X.drop('FareBin', axis=1)
'''
dummies = pd.get_dummies(X['Embarked'], prefix='Embarked', drop_first= True)
X= pd.concat([X, dummies], axis=1)

#Removed column already encoded
X = X.drop('Embarked', axis=1)
X.info()
'''
#correlation matrix
corrmat = Titanic_Train[['Survived','Pclass', 'Sex','Age','Embarked','Is_Married_Fem','Has_Family']].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Survived')['Survived'].index
cm = np.corrcoef(Titanic_Train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
'''

#PCA Feature reduction
#from sklearn.decomposition import PCA, KernelPCA
#pca = PCA(n_components=0.99, whiten=True)
#X = pca.fit_transform(X)

#kpca = KernelPCA(kernel='rbf',gamma=5, n_components=1)
#X = kpca.fit_transform(X)

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#lda = LinearDiscriminantAnalysis(n_components=1)
#X = lda.fit(X, y).transform(X)


#Train test split NO NEED TO SPLIT FOR CROSS VALIDATION
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=0)

#Apply scaling using Standardization 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X =  sc.fit_transform(X)
X.loc[:,['Age','Fare','Has_Family']] = sc.fit_transform(X.loc[:,['Age','Fare','Has_Family']])
#X.loc[:,['Age']] = sc.fit_transform(X.loc[:,['Age']])
#X_test.loc[:,['Age','Fare']] = sc.transform(X_test.loc[:,['Age', 'Fare']])
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

"""
#Apply scaling using Normalization
from sklearn.preprocessing import Normalizer
sc = Normalizer()
#X =  sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

"""
#apply Random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =100,criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

#Apply boosting
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(random_state=0)
classifier.fit(X_train, y_train)
"""
#Apply xgboost
import xgboost as xgb


"""
param = {
    'eta': 0.4, 
    'max_depth': 15,  
    'gamma': 0.4,
    'min_child_weight': 5,
    'objective': 'multi:softmax',  
    'num_class': 2,
    'eval_metric':'error'} 
"""
param = {
    'eta': 0.15, 
    'max_depth': 7,  
    'gamma': 0.2,
    'min_child_weight': 3,
    'max_features':0.7,
    'min_samples_leaf':0.6,
    'objective': 'multi:softmax',  
    'num_class': 2,
    'n_jobs':0,
    'random_state':42,
    "colsample_bytree":0.6
     } 
steps = 30  # The number of training iterations
#model = xgb.train(param, X_train, steps)
"""
model = xgb.XGBClassifier(objective='binary:logistic', eta=0.05, 
                          gamma=0.2,random_state=42, n_estimators=100, max_depth=7,
                          min_child_weight=3, max_features=0.7, 
                          min_samples_leaf=0.6, importance_type='gain',
                          learning_rate=0.5, colsample_bytree=0.6)
"""
from sklearn.model_selection import KFold, cross_val_score
#Create k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state =1)
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

#apply Random forest classifier
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators =500,criterion='entropy', random_state=0)

score = cross_val_score(model, X, y, cv=kf).mean()
print(score)
#fit model
#model.fit(X_train, y_train)

#Predict test set result
#y_pred = model.predict(X_test)


#Making confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(accuracy_score(y_test, y_pred))

#Get Accuracy score


#Rename 
#Estimated_Cases.rename(columns={"YEAR (CODE)":"Year", "REGION (DISPLAY)":"Region","COUNTRY (DISPLAY)":"Country",'Numeric':'Estimated_Cases'}, inplace=True)

#Drop columns
#Estimated_Cases.drop(Estimated_Cases.columns[[0,1,2,3,4,5,7,8,9,11,12,14,15,17,18,19]], axis=1, inplace=True)



#Movies['Title_year'] = [re.sub(r'.*\((.*)\)', r'\1', x) for x in Movies['Title']]

#Estimated = pd.merge(Estimated_Cases, Estimated_Deaths, how='inner', on = ['Year','Country']).fillna(0)




#Malaria_csvfile = Path+'\Malaria_Merged_Dataset.csv'

#Estimated_Confirmed_pop.to_csv(Malaria_csvfile, mode='w', index=False)    

