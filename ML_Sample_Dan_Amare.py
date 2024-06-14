from datetime import datetime
from logging import getLogger
import csv
import os
import numpy as np
import pandas as pd
import re as re
import subprocess as sb
import sys 
import cx_Oracle
import platform
import shutil
from datetime import date
import time
import dateutil
from dateutil import parser
import openpyxl
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import tensorflow as tf





#This is machine learning sample script to predict risk levels based on relevant features.
#THIS HAS NOT BEEN IMPLMENTED, only used for demo purposes.
#v1.1 Random Forest, Visualize feature importance
#v1.2 Change features to ordinal categories, with priority levels
#v1.3   Try with XGBOOST --> Better
#v1.4 Try with Neural Networks Regression --> Not good
#v1.5 NN Classifier -->Not better




logger = getLogger('return')


    
def print_log(LOG_FILE, print_txt, target): 
    if target == 1:
        print(print_txt) 
    with open(LOG_FILE, 'a') as f:
        f.write(print_txt)
        f.write('\n') 

def train_model(data_file_data, rf_priority):
    # Create mapper
    scale_mapper = {"Low":0,
        "Medium":1,
        "High":2, "Very High":3}
    # Replace feature values with scale
    #dataframe["Score"].replace(scale_mapper)       
    data_file_data['RiskLevel'] =  data_file_data['RiskLevel'].replace(scale_mapper) 
    rf_priority['PRIORITY'] =  rf_priority['PRIORITY'].replace(scale_mapper) 
    rf_priority = rf_priority.drop([0,1,2],axis=0)
    #print(rf_priority.head(7))
    #print(rf_priority.iloc[7,0])
    #exit(0)
    #replace boolean with Y/N
    #replace_txt = {'Yes':'Y','No':'N'}
    #data_file_data = data_file_data.replace(replace_txt,regex=False)
    #Replace None Yes/No with NA
    replace_txt = {'Yes': 1,
                   'No': -0.1,
                   'Not Available':0,
                   'Not Applicable':0,
                   'N/A': 0}
    data_file_data = data_file_data.replace(replace_txt,regex=False)
    #print(data_file_data.head())
    df_awarded = data_file_data[data_file_data['StatusCd']=='AWARDED']
    df_preaward = data_file_data[data_file_data['StatusCd']!='AWARDED']
    # Drop the 'Id' column as it's just an identifier
    df_awarded=df_awarded.drop(['GrntId','StatusCd'], axis=1)
    df_preaward=df_preaward.drop(['GrntId','StatusCd'], axis=1)
    #df_awarded.drop(['GrntId','StatusCd'], axis=1, inplace=True)
    #df_preaward.drop(['GrntId','StatusCd'], axis=1, inplace=True)
    """
    
    # One-hot encode the categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features_a = encoder.fit_transform(df_awarded.drop('RiskLevel', axis=1))
    encoded_features_p = encoder.fit_transform(df_preaward.drop('RiskLevel', axis=1))

    # Create a new DataFrame with the encoded features
    feature_names = encoder.get_feature_names_out(input_features=df_awarded.columns[1:])
    df_encoded_a = pd.DataFrame(encoded_features_a, columns=feature_names)
    df_encoded_p = pd.DataFrame(encoded_features_p, columns=feature_names)

    # Add the target column back to the encoded DataFrame
    df_encoded_a['RiskLevel'] = df_awarded['RiskLevel'].values
    df_encoded_p['RiskLevel'] = df_preaward['RiskLevel'].values
    
    #print(df_encoded_a.head())
    #exit(0)
    # Split the dataset into training and testing sets
    X_train = df_encoded_a.drop('RiskLevel', axis=1)
    y_train = df_encoded_a['RiskLevel']
    X_test = df_encoded_p.drop('RiskLevel', axis=1)
    y_test = df_encoded_p['RiskLevel']
    """
    X_train = df_awarded.drop('RiskLevel', axis=1)
    y_train = df_awarded['RiskLevel']
    X_test = df_preaward.drop('RiskLevel', axis=1)
    y_test = df_preaward['RiskLevel']
    row_index = 0
    for columns in X_train.columns:
        #print(columns)
        X_train[columns]=X_train[columns]*rf_priority.iloc[row_index,0]
        X_test[columns]=X_test[columns]*rf_priority.iloc[row_index,0]
        row_index += 1

    """
    # Neural network model for regression
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)  # Single output neuron without activation for regression
        ])


    
    model.compile(optimizer='adam',
              loss='mean_squared_error',  # MSE is commonly used for regression tasks
              metrics=['mean_absolute_error'])  # MAE provides an intuitive error metric
    """
    # Neural network model Classifier
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(y_train.nunique(), activation='softmax')  # Assuming 'Risk' is a categorical variable
    #tf.keras.layers.Dense([1,2,3,4], activation='softmax')  # Assuming 'Risk' is a categorical variable
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=2)

    print(f"Test accuracy: {val_acc}")

    # Predict and calculate accuracy (optional)
    # Convert predictions to labels
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f'Neural Network accuracy: {accuracy}')


    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the Random Forest classifier
    #Best estimators using GridSearch
    #'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    #rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    #rf_classifier = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42)
    #xgboost
    """"
    rf_classifier = XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')
    
    rf_classifier.fit(X_train, y_train)

    # Make predictions and calculate the accuracy
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model accuracy on the test set:", accuracy)
    
    # Assuming best_rf is your best estimator from GridSearchCV

    # Get feature importances
    best_rf = rf_classifier
    feature_importances = best_rf.feature_importances_

    # Sort the feature importances in descending order and get the indices
    sorted_indices = np.argsort(feature_importances)[::-1]

    # Number of top features to select, you can adjust this number
    top_n = 20

    # Select the top n feature names and their importance scores
    #top_feature_names = df_encoded_a.columns[sorted_indices][:top_n]
    top_feature_names = df_awarded.columns[sorted_indices][:top_n]
    top_feature_importances = feature_importances[sorted_indices][:top_n]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_feature_importances[::-1], align='center')
    plt.yticks(range(top_n), top_feature_names[::-1])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top {} Most Important Features'.format(top_n))
    plt.show()


    
    # Create features and target
    features = data_file_data.iloc[:,2:]
    print(features.head())
    # Create one-hot encoder
    one_hot = LabelBinarizer()
    #one_hot = MultiLabelBinarizer()
    # One-hot encode feature
    one_hot.fit_transform(features[:,0])
    print(one_hot.classes_)



    target = data_file_data.iloc[:,1]
    print(target.head())

    """

    return

def main( v_data_file):
#def main():
    #examp

    DEMO_MAX = 2000
    num_rows = 0
    VERSION_CODE = 1.0
    #VERSION_CONFIG = 20
    WORKING_DIRECTORY = os.getcwd()
    #print('Imported Index directory is '+ INDEX_DIRECTORY)
    #print('Imported CACHE directory is '+ IRSX_CACHE_DIRECTORY)
    #print('Imported XML directory is '+ WORKING_DIRECTORY)
    today = datetime.now().strftime("%m_%d_%Y_%H_%M")
    
    
    data_file_curr = v_data_file
    data_file_ext = os.path.splitext(data_file_curr)[-1].lower()

    data_file_log = 'ML_log_'+today+'.txt'
    #CONFIG_FILE = os.path.join(WORKING_DIRECTORY, 'Config','Data_file_structs.xls') 
    CONFIG_FILE = os.path.join(WORKING_DIRECTORY, 'Config','ML_data_struct.xls') 
    DATA_FILE = os.path.join(WORKING_DIRECTORY, 'Data',data_file_curr) 
    LOG_FILE = os.path.join(WORKING_DIRECTORY, 'Logs',data_file_log) 
    print_txt='Data load script version: %s\n'%(VERSION_CODE)
    #print_txt=print_txt + 'Config file version: %s'%(VERSION_CONFIG)    
    print_log(LOG_FILE, print_txt,1)    
    
    if not os.path.exists(DATA_FILE):
        print_txt='Error: Input file was not found: %s '%(DATA_FILE)
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)
    else:
        print_txt = "Input file: %s" %(DATA_FILE)         
        print_log(LOG_FILE, print_txt,1)
        
        
    #data_name_curr = v_data_name
    try:
        data_file_headers = pd.read_excel(CONFIG_FILE, names=['FIELDS','PRIORITY'],sheet_name = 'ML',header=None, index_col=False)

    except Exception as err:
            print_txt = 'Error reading config file. Expected column missing or not in expected location.\n'
            print_txt = print_txt+'%s'%(err)        
            print_log(LOG_FILE, print_txt,1)
            sys.exit(1) 

    data_file_fields = data_file_headers['FIELDS'].T.tolist()
    #print(data_file_fields)
    #return
    rf_priority = data_file_headers[['PRIORITY']]
    #print(data_file_cols)
    #return
    
    #print (file_headers.loc['COLS'].tolist())
    #usecols=lambda c: c in set(data_file_index) for changing index to column names
    #USAS file no longer being used, API version implemented in API script
    data_file_sheet = {'REVIEW': 'New and Recompete',
                        'MONREF': 'NC_OCRO',
                        'FIELDPRINT': 0
                            }
    data_file_cols ={'IPERA':'A:Z',
                     'REVIEW': 'C,N,DN'}


    #If TrueScreen, skip 4 report title rows
    #skip_num_rows = 4 if data_name_curr == 'TRUESCREEN' else 0
    try:
        data_file_data = pd.read_excel(DATA_FILE,
                                               names=data_file_fields,
                                               sheet_name = 0,
                                               header=0, 
                                               #skiprows= skip_num_rows,
                                               usecols='G,L,AH,AM:DR',
                                               dtype=str,
                                               keep_default_na=False,
                                               engine='openpyxl',index_col=False)
    except Exception as err:
        print_txt = 'Error reading file. Expected column missing or not in expected location.\n'
        print_txt = print_txt+'%s'%(err)        
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)        

    #data_file_data.to_csv(LOG_FILE, index=False)
    #print (data_file_data.columns)
    #return
    #Remove null records
    data_file_data.columns = data_file_fields
    data_file_data = data_file_data.replace(r'^\s*$', np.nan, regex=True)
    data_file_data = data_file_data.dropna(axis=0, how='all')
    #print(data_file_data.columns)
    #print(data_file_data.head())
    train_model(data_file_data, rf_priority)

    #load_data(v_db_env,data_name_curr, data_file_fields, data_file_data, LOG_FILE)


if __name__ == '__main__':
    
    if len(sys.argv) != 2:        
        #raise ValueError('FAILED -- Invalid job execution format.')
        print('Error: Invalid job execution format.\n'+
                    'Please using following format to execute the model.\n'+
                            'ML_Pilot.py [Data file name].\n'+
                            'Example: ML_Pilot.py '+"'"+'Applications Pivot Report.xlsx'+"'")
        sys.exit(1)
    v_data_file = sys.argv[1]
    #v_credentials = sys.argv[2]
    

    main(v_data_file)