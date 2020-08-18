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

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, make_scorer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline


# file path load
Path=r"C:\Users\damar\Documents\Python Scripts\Housing"
df_train = pd.read_csv(Path+r'\train.csv', sep=',',engine='python')
df_test = pd.read_csv(Path+r'\test.csv', sep=',',engine='python')
df_train.head()
df_test.head()
df_train.columns

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

df_train.shape
#Delete outlier data
#df_df_all_data = df_df_all_data.drop(df_df_all_data[df_df_all_data['Id'] == 1299].index)
#df_df_all_data = df_df_all_data.drop(df_df_all_data[df_df_all_data['Id'] == 524].index)

df_train.shape
df_test.shape
#Log transformation to dependent variable
#applying log transformation because of positive skewness
df_train['SalePrice'] = np.log(df_train['SalePrice'])


#missing data, merge data first
df_all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
df_saleprice = df_train["SalePrice"]
df_all_data.drop(["SalePrice"], axis = 1, inplace= True)
# Drop Id column
df_all_data.drop("Id", axis = 1, inplace = True)
df_all_data.shape

total = df_all_data.isnull().sum().sort_values(ascending=False)
percent = (df_all_data.isnull().sum()/df_all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
#df_all_data = df_all_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
#df_all_data = df_all_data.drop(df_all_data.loc[df_all_data['Electrical'].isnull()].index)


# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"
df_all_data.loc[:, "Alley"] = df_all_data.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
df_all_data.loc[:, "BedroomAbvGr"] = df_all_data.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
df_all_data.loc[:, "BsmtQual"] = df_all_data.loc[:, "BsmtQual"].fillna("No")
df_all_data.loc[:, "BsmtCond"] = df_all_data.loc[:, "BsmtCond"].fillna("No")
df_all_data.loc[:, "BsmtExposure"] = df_all_data.loc[:, "BsmtExposure"].fillna("No")
df_all_data.loc[:, "BsmtFinType1"] = df_all_data.loc[:, "BsmtFinType1"].fillna("No")
df_all_data.loc[:, "BsmtFinType2"] = df_all_data.loc[:, "BsmtFinType2"].fillna("No")
df_all_data.loc[:, "BsmtFullBath"] = df_all_data.loc[:, "BsmtFullBath"].fillna(0)
df_all_data.loc[:, "BsmtHalfBath"] = df_all_data.loc[:, "BsmtHalfBath"].fillna(0)
df_all_data.loc[:, "BsmtUnfSF"] = df_all_data.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
df_all_data.loc[:, "CentralAir"] = df_all_data.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
df_all_data.loc[:, "Condition1"] = df_all_data.loc[:, "Condition1"].fillna("Norm")
df_all_data.loc[:, "Condition2"] = df_all_data.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
df_all_data.loc[:, "EnclosedPorch"] = df_all_data.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
df_all_data.loc[:, "ExterCond"] = df_all_data.loc[:, "ExterCond"].fillna("TA")
df_all_data.loc[:, "ExterQual"] = df_all_data.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
df_all_data.loc[:, "Fence"] = df_all_data.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
df_all_data.loc[:, "FireplaceQu"] = df_all_data.loc[:, "FireplaceQu"].fillna("No")
df_all_data.loc[:, "Fireplaces"] = df_all_data.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
df_all_data.loc[:, "Functional"] = df_all_data.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
df_all_data.loc[:, "GarageType"] = df_all_data.loc[:, "GarageType"].fillna("No")
df_all_data.loc[:, "GarageFinish"] = df_all_data.loc[:, "GarageFinish"].fillna("No")
df_all_data.loc[:, "GarageQual"] = df_all_data.loc[:, "GarageQual"].fillna("No")
df_all_data.loc[:, "GarageCond"] = df_all_data.loc[:, "GarageCond"].fillna("No")
df_all_data.loc[:, "GarageArea"] = df_all_data.loc[:, "GarageArea"].fillna(0)
df_all_data.loc[:, "GarageCars"] = df_all_data.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
df_all_data.loc[:, "HalfBath"] = df_all_data.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
df_all_data.loc[:, "HeatingQC"] = df_all_data.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
df_all_data.loc[:, "KitchenAbvGr"] = df_all_data.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
df_all_data.loc[:, "KitchenQual"] = df_all_data.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
df_all_data.loc[:, "LotFrontage"] = df_all_data.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
df_all_data.loc[:, "LotShape"] = df_all_data.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
df_all_data.loc[:, "MasVnrType"] = df_all_data.loc[:, "MasVnrType"].fillna("None")
df_all_data.loc[:, "MasVnrArea"] = df_all_data.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
df_all_data.loc[:, "MiscFeature"] = df_all_data.loc[:, "MiscFeature"].fillna("No")
df_all_data.loc[:, "MiscVal"] = df_all_data.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
df_all_data.loc[:, "OpenPorchSF"] = df_all_data.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
df_all_data.loc[:, "PavedDrive"] = df_all_data.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
df_all_data.loc[:, "PoolQC"] = df_all_data.loc[:, "PoolQC"].fillna("No")
df_all_data.loc[:, "PoolArea"] = df_all_data.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
df_all_data.loc[:, "SaleCondition"] = df_all_data.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
df_all_data.loc[:, "ScreenPorch"] = df_all_data.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
df_all_data.loc[:, "TotRmsAbvGrd"] = df_all_data.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
df_all_data.loc[:, "Utilities"] = df_all_data.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
df_all_data.loc[:, "WoodDeckSF"] = df_all_data.loc[:, "WoodDeckSF"].fillna(0)


# Some numerical features are actually really categories
df_all_data = df_all_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })


#Encode some categorical features as ordered numbers when there is information in the order
df_all_data = df_all_data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )


# Create new features
# 1* Simplifications of existing features
df_all_data["SimplOverallQual"] = df_all_data.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
df_all_data["SimplOverallCond"] = df_all_data.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
df_all_data["SimplPoolQC"] = df_all_data.PoolQC.replace({1 : 1, 2 : 1, # average
                                             3 : 2, 4 : 2 # good
                                            })
df_all_data["SimplGarageCond"] = df_all_data.GarageCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
df_all_data["SimplGarageQual"] = df_all_data.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
df_all_data["SimplFireplaceQu"] = df_all_data.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
df_all_data["SimplFireplaceQu"] = df_all_data.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
df_all_data["SimplFunctional"] = df_all_data.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })
df_all_data["SimplKitchenQual"] = df_all_data.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
df_all_data["SimplHeatingQC"] = df_all_data.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
df_all_data["SimplBsmtFinType1"] = df_all_data.BsmtFinType1.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
df_all_data["SimplBsmtFinType2"] = df_all_data.BsmtFinType2.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
df_all_data["SimplBsmtCond"] = df_all_data.BsmtCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
df_all_data["SimplBsmtQual"] = df_all_data.BsmtQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
df_all_data["SimplExterCond"] = df_all_data.ExterCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
df_all_data["SimplExterQual"] = df_all_data.ExterQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })


#for column in df_all_data.columns[df_all_data.isna().any()].tolist():
    #df_all_data = df_all_data.drop(df_all_data.loc[df_all_data[column].isnull()].index)
    #print(df_all_data.loc[df_all_data[column].isnull()].index)

df_all_data.isnull().sum().max() #just checking that there's no missing data missing...
#df_all_data.fillna(0)
df_all_data.shape
#print(df_all_data.iloc[524:525, :])
#print(df_saleprice.iloc[523:525])

#log transformation to some indpendent variables
#data transformation
df_all_data['GrLivArea'] = np.log(df_all_data['GrLivArea'])

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_all_data['HasBsmt'] = pd.Series(len(df_all_data['TotalBsmtSF']), index=df_all_data.index)
df_all_data['HasBsmt'] = 0 
df_all_data.loc[df_all_data['TotalBsmtSF']>0,'HasBsmt'] = 1

#transform data for nonzero basement square feet
df_all_data.loc[df_all_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_all_data['TotalBsmtSF'])



# Differentiate numerical featbures (minus the target) and categorical features
objList = df_all_data.select_dtypes(include = ["object"]).columns
numList = df_all_data.select_dtypes(exclude = ["object"]).columns
#numList = numList.drop("SalePrice")
print("Numerical features : " + str(len(numList)))
print(numList)
print("Categorical features : " + str(len(objList)))
print(objList)
df_all_data_num = df_all_data[numList]
df_all_data_cat = df_all_data[objList]

# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train : " + str(df_all_data[numList].isnull().values.sum()))
df_all_data[numList] = df_all_data[numList].fillna(df_all_data[numList].median())
print("Remaining NAs for numerical features in train : " + str(df_all_data[numList].isnull().values.sum()))


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    i = 0
    for each in cols:
        #print (each)
        dummies = pd.get_dummies(df[each], prefix=each, drop_first= True)
        if i == 0: 
            print (dummies)
            i = i + 1
        df = pd.concat([df, dummies], axis=1)
    return df

#One hot encoding done
df_all_data = one_hot(df_all_data, objList)

#Dropping duplicates columns if any
df_all_data = df_all_data.loc[:,~df_all_data.columns.duplicated()]
df_all_data.shape

#Dropping the original columns that has data type object 
df_all_data.drop(objList, axis=1, inplace=True)
df_all_data.shape


#Check no objList left
objList = df_all_data.select_dtypes(include = "object").columns
print (objList)

'''
???
#Label Encoding for object to numeric conversion - Option 1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objList:
    df_all_data[feat] = le.fit_transform(df_all_data[feat].astype(str))

print (df_all_data.shape)
'''
#Save for modelling, df_all_data-test split
df_train = df_all_data.iloc[:1460,:]  

#Save for submission
X_test_sub = df_all_data.iloc[1460 :,:]  

#print(df_saleprice.iloc[522:525])
#print(df_saleprice.iloc[1296:1299])

df_train["SalePrice"] = df_saleprice


X = df_train.drop(["SalePrice"], axis = 1)
y = df_train["SalePrice"]

#Save for submission
#X_test = df_test

# Partition the dataset in df_all_data + validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("X_df_all_data : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_df_all_data : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


# Standardize numerical features
stdSc = StandardScaler()
X_train.loc[:, numList] = stdSc.fit_transform(X_train.loc[:, numList])
X_test.loc[:, numList] = stdSc.transform(X_test.loc[:, numList])
X_test_sub.loc[:, numList] = stdSc.transform(X_test_sub.loc[:, numList])



#*****
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
steps = 30  # The number of df_all_dataing iterations
#model = xgb.df_all_data(param, X_df_all_data, steps)

from sklearn.model_selection import KFold, cross_val_score
#Create k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state =1)
"""
objective='reg:logistic', eta=0.05, 
                          gamma=0.4,random_state=0, n_estimators=100, max_depth=15,
                          min_child_weight=1, max_features=0.4, 
                          base_score = 0.5,
                          min_samples_leaf=0.6, importance_type='gain',
                          learning_rate=0.1, colsample_bytree=0.3
"""

# Define error measure for official scoring : RMSE
scorer = make_scorer(mean_squared_error, greater_is_better = False)
'''
def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)
'''
def rmse_cv(model, data):
    rmse= np.sqrt(-cross_val_score(model, data, y_test, scoring = scorer, cv = 10))
    return(rmse)
    



#Apply XGBoost Regressor

import xgboost as xgb
from xgboost import plot_importance

'''
model = xgb.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
       learning_rate=0.1, max_delta_step=0, max_depth=2,
       min_child_weight=1, missing=None, n_estimators=360, nthread=-1,
       objective='reg:linear', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
#rmsle_xgb = test_model(reg_xgb)

#score = cross_val_score(model, X_df_all_data, y_df_all_data, cv=kf).mean()
#print(score)
'''
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse',
                          init=None, learning_rate=0.05, loss='ls', max_depth=3,
                          max_features='sqrt', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=9, min_samples_split=8,
                          min_weight_fraction_leaf=0.0, n_estimators=1250,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=0.8, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)

#Score X_test data
#score = rmse_cv(model, X_test).mean()
#print ('Rmse score is: '+ str(score))


#fit model for submission
model.fit(X_train, y_train)

#Predict test set result
y_pred = model.predict(X_test_sub)

#Transform back log values
y_pred = np.exp(y_pred)

header = ['SalePrice']

Housing_Pred = pd.concat([df_test,pd.DataFrame(y_pred, columns=header)],axis=1)

df_test.head()

Housing_csvfile = Path+'\Housing_submission2.csv'

Housing_Pred[['Id', 'SalePrice']].to_csv(Housing_csvfile, mode='w', index=False) 

