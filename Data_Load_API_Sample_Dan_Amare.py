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
import requests
from requests.exceptions import HTTPError
from pandas import json_normalize
import zipfile
import io


#This Python script essentially obtains and imports data made available to public by federal agencies as GSA, IRS, etc.
# Python's requests library is used to pull data from various APIs, transform as needed and important into oracle tables.
# In some cases, internal lookup data is used to limit the data requested prior to importing.
# In other cases, data is transformed, aggregated, engineered based on requirements.
# THIS SCRIPT IS FOR DEMO ONLY, IT DOES NOT WORK AS ESSENTIAL PARAMETERS, CONFIG files, folders are made inaccessible.
# Entire script implemented by Dagnaw Amare, 2022-2024. 
 



logger = getLogger('return')

def get_registrations_data(myAPIKey,connection, cursor,ra_table,data_file_fields,data_file_cols,LOG_FILE):


    baseURL = "https://api.sam.gov/entity-information/v2/entities?"
    baseURL = baseURL + "&api_key=" + myAPIKey
    #myAPIKey = "Bvp9zNegY3y5tU7C5DQZlDsn1Pw5V9I58bEubE8Q"
    qterms = "GSA"
        
        #Collect organizations in Monitoring/Applications universe
        #sql = """select distinct uei_nbr,name_upper ORG_NAME
        #    from ares.mv_daily_app_universe mu, organizations o where mu.org_id = o.id """
    sql = """select distinct uei_nbr,name_upper ORG_NAME
            from ares.mv_daily_app_universe mu, organizations o where mu.org_id = o.id 
            and uei_nbr is not null
            """

       #cngrants_duns = pd.read_sql(sql, con=connection)   
    cursor.execute(sql)
    cngrants = pd.DataFrame(cursor.fetchall(),columns=['UEI_NBR','ORG_NAME'])
    #cngrants_uei = cngrants[cngrants['UEI_NBR'].notnull()]['UEI_NBR']
    #cngrants_oname = cngrants[cngrants['UEI_NBR'].isnull()]['ORG_NAME']
    cngrants_uei = cngrants[cngrants['UEI_NBR'].notnull()]
    #cngrants_oname = cngrants[cngrants['UEI_NBR'].isnull()]
    #print(cngrants_uei.head())
    #print(cngrants_oname.head())
        
    print_txt = 'Number of Orgs in Application Universe: %s'%cursor.rowcount
    print_log(LOG_FILE, print_txt,1)

    #cngrants_duns.columns = [x[0] for x in cursor.description]
    #Collect orgs already loaded
    sql_loaded = """select DISTINCT UNIQUE_ENTITY_ID_SAM, 
                        INITIAL_REGISTRATION_DT, EXPIRATION_DATE, 
                        LAST_UPDATE_DT, ACTIVATION_DATE, 
                        LEGAL_BUSINESS_NM from """+ra_table
    cursor.execute(sql_loaded)        
    ra_sam_reg = pd.DataFrame(cursor.fetchall(), columns=['UNIQUE_ENTITY_ID_SAM_TBL', 
                        'INITIAL_REGISTRATION_DT_TBL', 'EXPIRATION_DATE_TBL', 
                        'LAST_UPDATE_DT_TBL', 'ACTIVATION_DATE_TBL', 
                        'LEGAL_BUSINESS_NM_TBL'])
    print_txt='Organizations previously loaded: %s'%cursor.rowcount
    print_log(LOG_FILE, print_txt,1)
    #print(cursor.description)
    #return
    #ra_usas_duns_fy.columns = [x[0] for x in cursor.description]

    #limit to orgs in CN_GRANTS using non null UEI
    cngrants = pd.DataFrame([], columns = data_file_fields)
    search_key = ['UEI','ONAME']
    uei_set = []
    oname_set = []
    
    #print(range(len(cngrants_uei['UEI_NBR'])))
    for i in range(len(cngrants_uei['UEI_NBR'])):
        uei_set.append(cngrants_uei['UEI_NBR'][i])
        if (((i+1)%50 == 0)  or (i==(len(cngrants_uei['UEI_NBR'])-1))):
            #Limit to 100 uei per transaction, less than 2k url limit
            uei_str = '~'.join(uei_set)
            entity_recs, rec_found = get_entity_data(baseURL,data_file_cols, data_file_fields, uei_str, search_key[0], LOG_FILE)
            if rec_found:
                cngrants = pd.concat([cngrants, entity_recs],axis=0)
            uei_set = []

    #print ('Length of Oname Series" %s'%len(cngrants_oname['ORG_NAME']))           
    """
    count = 0
    for oname in cngrants_oname['ORG_NAME']:        
        oname_set.append(oname)
        #limit to 30 orgs per transaction, less than 2k url limit
        if (((count+1)%30 == 0)  or (count==(len(cngrants_oname['ORG_NAME'])-1))):
            #oname_str = '\"~\"'.join(oname_set)
            oname_str = '%22~%22'.join(oname_set)
            entity_recs, rec_found = get_entity_data(baseURL,data_file_cols, data_file_fields, oname_str, search_key[1], LOG_FILE)
            if rec_found:
                cngrants = pd.concat([cngrants, entity_recs],axis=0)        
            oname_set = []
        count += 1
    """
    data_file_data = cngrants
    #filter out data already loaded
    sam_new = pd.merge(data_file_data,ra_sam_reg,
                    left_on=['UNIQUE_ENTITY_ID_SAM', 
                        'INITIAL_REGISTRATION_DT', 'EXPIRATION_DATE', 
                        'LAST_UPDATE_DT', 'ACTIVATION_DATE', 
                        'LEGAL_BUSINESS_NM'],
                    right_on=['UNIQUE_ENTITY_ID_SAM_TBL', 
                        'INITIAL_REGISTRATION_DT_TBL', 'EXPIRATION_DATE_TBL', 
                        'LAST_UPDATE_DT_TBL', 'ACTIVATION_DATE_TBL', 
                        'LEGAL_BUSINESS_NM_TBL'],
                     how='left')
    #ra_sam_reg.to_csv("SAM_Reg_Previous.csv")    
        
    sam_new = sam_new[sam_new['LEGAL_BUSINESS_NM_TBL'].isnull()]
    data_file_data = sam_new.drop(['UNIQUE_ENTITY_ID_SAM_TBL', 
                        'INITIAL_REGISTRATION_DT_TBL', 'EXPIRATION_DATE_TBL', 
                        'LAST_UPDATE_DT_TBL', 'ACTIVATION_DATE_TBL', 
                        'LEGAL_BUSINESS_NM_TBL'], axis=1)
        
    print_txt ='New SAM Registration records to be loaded: %s'%(len(data_file_data.index))        
    print_log(LOG_FILE, print_txt,1)

    return data_file_data

def get_entity_data(baseURL,data_file_cols, data_file_fields, key_val, search_key, LOG_FILE):
    
    if (search_key == 'UEI'):
        parameter = "&ueiSAM=["+key_val+"]"
    else: 
        key_val = key_val.replace('&', '')
        parameter = "&legalBusinessName=[%22"+key_val+"%22]"
        
    #queryURL = baseURL + qterms + "&api_key=" + myAPIKey
    queryURL = baseURL + parameter
    search_uei = len(key_val.split('~'))
    #print(queryURL)
    """
    try:
        response = requests.get(queryURL)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  
    except Exception as err:
        print(f'Other error occurred: {err}')  
    except (KeyError, TypeError) as e:
        # logger.debug(e)
        print(f'No rec found for %s: {err}'%key_val)  
        return ([], False)
        #pass
    """
    #else:
        #print('Successfully retreived data from ' + queryURL)
    data_recs = []
    data_list = []
    data_dict = {}
    new_results = True
    page = 0
    while new_results:
        entities = requests.get(queryURL + f"&page={page}&size=10").json()
        new_results = entities.get("entityData", [])
        data_recs.extend(new_results)
        page += 1
    
    #extract fields and records into list
    for i in range(len(data_recs)):
        data_reg = data_recs[i]['entityRegistration']
        data_ent = data_recs[i]['coreData']['entityInformation']
        data_addr = data_recs[i]['coreData']['physicalAddress']
        data_fin = data_recs[i]['coreData']['financialInformation']
        #concatenate all dicts
        data_dict = {**data_reg, **data_ent, **data_addr, **data_fin}
        data_list.append(data_dict)
        #exit(0)
    if (len(data_recs) > 0):
        print_txt='Number of Entity Registration found: %s/%s'%(len(data_recs),search_uei)
        print_log(LOG_FILE, print_txt,1)
    else:    
        #No records found so return
        #print(f'No rec found for %s: '%key_val)
        print_txt='No Registration record found for %s Entities.'%(search_uei)  
        print_log(LOG_FILE, print_txt,1)
        return [],False
    #Convert list of dicts into dataframe
    df = pd.DataFrame(data_list)
    #retrieve only fields needed from config file
    df = df[data_file_cols]
    #replace SAM field names with database field names
    df.columns = data_file_fields
    
    
    #df = json_normalize(data_rec)
    #today = datetime.now().strftime("%m_%d_%Y_%H_%M")
    
    #data_file_output = 'Reg_api_data_'+today+'.csv'
    #DATA_API = os.path.join(os.getcwd(), 'Data',data_file_output) 
    #df.to_csv(DATA_API,index=False)
    #print(df.head())
    
    return (df, True)

def get_exclusions_data(myAPIKey,connection, cursor,ra_table,data_file_fields,data_file_cols,LOG_FILE):


    #baseURL = "https://api.sam.gov/entity-information/v2/entities?"
    baseURL = "https://api.sam.gov/entity-information/v2/exclusions?"
    baseURL = baseURL + "api_key=" + myAPIKey
    #myAPIKey = "Bvp9zNegY3y5tU7C5DQZlDsn1Pw5V9I58bEubE8Q"
    
        
    #Collect organizations in Monitoring/Applications universe
    sql = """select distinct uei_nbr,name_upper ORG_NAME
            from ares.mv_daily_app_universe mu, organizations o where mu.org_id = o.id 
            and uei_nbr is not null"""
    #No longer using Org Name as search criteria, not very accurate
    """ union
            select distinct uei_nbr,name_upper ORG_NAME
            from ares.mv_daily_app_universe mu, organizations o where mu.org_id = o.id 
            and uei_nbr is null
            and rownum < 100"""

    cursor.execute(sql)
    cngrants = pd.DataFrame(cursor.fetchall(),columns=['UEI_NBR','ORG_NAME'])
    #cngrants_uei = cngrants[cngrants['UEI_NBR'].notnull()]['UEI_NBR']
    #cngrants_oname = cngrants[cngrants['UEI_NBR'].isnull()]['ORG_NAME']
    cngrants_uei = cngrants[cngrants['UEI_NBR'].notnull()]
    #cngrants_oname = cngrants[cngrants['UEI_NBR'].isnull()]
    #print(cngrants_uei.head())
    #print(cngrants_oname.head())
        
    print_txt = 'Number of Orgs in Application Universe: %s'%cursor.rowcount
    print_log(LOG_FILE, print_txt,1)

    #cngrants_duns.columns = [x[0] for x in cursor.description]
        
    #Get authorized staff in applications universe
    sql = """SELECT distinct AR.LAST_NM || ', ' || AR.FIRST_NM AS STAFF_NAME
                FROM CN_GRANTS G, CN_PEOPLE AR, ARES.MV_DAILY_APP_UNIVERSE MU 
                WHERE G.AUTH_REP_ID = AR.PER_ID AND g.grnt_id = mu.grnt_id 
                """
        
    cursor.execute(sql)
    cngrants_auth = pd.DataFrame(cursor.fetchall(),columns=['STAFF_NAME'])
    print_txt = 'Number of Authorized Staff in Application Universe: %s'%cursor.rowcount
    print_log(LOG_FILE, print_txt,1)
        
    #cngrants_duns.columns = [x[0] for x in cursor.description]
    #Collect orgs and staff already loaded
    sql_loaded = """select DISTINCT UNIQUE_ENTITY_ID, NAME,FIRST_NAME, MIDDLE, LAST_NAME,
                        TERMINATION_DATE FROM """+ra_table
    cursor.execute(sql_loaded)        
    ra_sam_excl = pd.DataFrame(cursor.fetchall(), columns=['UNIQUE_ENTITY_ID_TBL', 
                        'NAME_TBL', 'FIRST_NAME_TBL','MIDDLE_TBL','LAST_NAME_TBL', 'TERMINATION_DATE_TBL'])
    print_txt='Excluded Orgs and Authorized staff previously loaded: %s'%cursor.rowcount
    print_log(LOG_FILE, print_txt,1)
            
    #print(cursor.description)
    #return
    #ra_usas_duns_fy.columns = [x[0] for x in cursor.description]

    #limit to orgs in CN_GRANTS using non null UEI
    cngrants = pd.DataFrame([], columns = data_file_fields)
    search_key = ['UEI','ONAME','STAFF']
    uei_set = []
    oname_set = []
    staff_set = []
    count=0
    #print("Length is: %s"%(len(cngrants_uei['UEI_NBR'])))
    for uei in cngrants_uei['UEI_NBR']:
        uei_set.append(uei)
        #Limit to 100 uei per transaction, less than 2k url limit
        if (((count+1)%50 ==0) or (count == (len(cngrants_uei['UEI_NBR'])-1))):
            #print("UEI set is: %s"%uei_set)
            uei_str = '~'.join(uei_set)        
            excluded_recs, rec_found = get_excluded_data(baseURL,data_file_cols, data_file_fields, uei_str, search_key[0], LOG_FILE)
            if rec_found:
                cngrants = pd.concat([cngrants, excluded_recs],axis=0)
            uei_set = []
        count += 1
    
    #print ('Length of Oname Series" %s'%len(cngrants_oname['ORG_NAME']))               
    """count = 0
    for oname in cngrants_oname['ORG_NAME']:
        oname_set.append(oname)
        #limit to 30 orgs per transaction, less than 2k url limit
        if (((count+1)%30 == 0)  or (count==(len(cngrants_oname['ORG_NAME'])-1))):
            #print("Oname set is: %s"%oname_set)
            oname_str = '\"~\"'.join(oname_set)
            excluded_recs, rec_found = get_excluded_data(baseURL,data_file_cols, data_file_fields, oname_str, search_key[1], LOG_FILE)
            if rec_found:
                cngrants = pd.concat([cngrants, excluded_recs],axis=0)        
            oname_set = []
        count += 1
    """
    
    count = 0 
    for staff in cngrants_auth['STAFF_NAME']:
        full_name = "(%22"+staff.split(',')[1].strip() + "%22AND%22"+staff.split(',')[0].strip()+"%22)"
        #print(full_name)
        staff_set.append(full_name)
        #limit to 50 orgs per transaction, less than 2k url limit
        if (((count+1)%50 == 0)  or (count==(len(cngrants_auth['STAFF_NAME'])-1))):
            #print("Staff set is: %s"%staff_set)
            staff_str = '~'.join(staff_set)
            excluded_recs, rec_found = get_excluded_data(baseURL,data_file_cols, data_file_fields, staff_str, search_key[2], LOG_FILE)
            if rec_found:
                cngrants = pd.concat([cngrants, excluded_recs],axis=0)        
            staff_set = []
        count += 1

    data_file_data = cngrants
    #filter out data already loaded
    data_file_data['FIRST_NAME'] = data_file_data['FIRST_NAME'].str.upper()
    data_file_data['LAST_NAME'] = data_file_data['LAST_NAME'].str.upper()
    data_file_data['MIDDLE'] = data_file_data['MIDDLE'].str.upper()    
    data_file_data['NAME'] = data_file_data['NAME'].str.upper()
    
    #ra_sam_excl['FIRST_NAME_TBL'] = ra_sam_excl['FIRST_NAME_TBL'].str.upper()
    #ra_sam_excl['LAST_NAME_TBL'] = ra_sam_excl['LAST_NAME_TBL'].str.upper()
    sam_new = pd.merge(data_file_data,ra_sam_excl,
                    left_on=['UNIQUE_ENTITY_ID', 'NAME','FIRST_NAME', 'MIDDLE','LAST_NAME',
                        'TERMINATION_DATE'],
                    right_on=['UNIQUE_ENTITY_ID_TBL', 'NAME_TBL','FIRST_NAME_TBL','MIDDLE_TBL', 
                        'LAST_NAME_TBL' , 'TERMINATION_DATE_TBL'],
                     how='left')
            
    #sam_new = sam_new[sam_new['TERMINATION_DATE_TBL'].isnull()]
    sam_new = sam_new[sam_new['NAME_TBL'].isnull()]
    data_file_data = sam_new.drop(['UNIQUE_ENTITY_ID_TBL', 'NAME_TBL','FIRST_NAME_TBL', 'MIDDLE_TBL',
                        'LAST_NAME_TBL', 'TERMINATION_DATE_TBL'], axis=1)

    print_txt ='New SAM Exclusions records to be loaded: %s'%(len(data_file_data.index))        
    print_log(LOG_FILE, print_txt,1)

    return data_file_data

def get_excluded_data(baseURL,data_file_cols, data_file_fields, key_val, search_key, LOG_FILE):
    
    #ueiSAM="C111ATT311C8"
    #nameSAM="RIDE ON "
    #ueiSAM=key_val
    #nameSAM=key_val
    
    if (search_key == 'UEI'):
        parameter = "&classification=!INDIVIDUAL&ueiSAM=["+key_val+"]"
        classification = 'Entity'
    elif (search_key == 'ONAME'): 
        key_val = key_val.replace('&', '')
        parameter = "&classification=!INDIVIDUAL&exclusionName=[%22"+key_val+"%22]"
    else: 
        #parameter = "&classification=INDIVIDUAL&exclusionName=(%22"+first_name+"%22AND%22"+last_name+"%22)"
        parameter = "&classification=INDIVIDUAL&exclusionName=["+key_val+"]"
        classification = 'Individual'
        #print(parameter)
    #queryURL = baseURL + qterms + "&api_key=" + myAPIKey
    queryURL = baseURL + parameter
    search_uei = len(key_val.split('~'))
    
    #print(queryURL)
    #exit(0)
    """
    try:
        response = requests.get(queryURL)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  
    except Exception as err:
        print(f'Other error occurred: {err}')  
    except (KeyError, TypeError) as e:
        # logger.debug(e)
        print(f'No rec found for %s: {err}'%key_val)  
        return ([], False)
        #pass
    """
    data_recs = []
    data_list = []
    data_dict = {}
    new_results = True
    page = 0
    while new_results:
        exclusions = requests.get(queryURL + f"&page={page}&size=10").json()
        new_results = exclusions.get("excludedEntity", [])
        data_recs.extend(new_results)
        page += 1
    
    #print('Length of data recs is: %s'%len(data_recs))
    #print('Number of pages is: %s'%(page-1))
    #print('First data rec is: %s'%data_recs[0])
    #exit(0)
    #extract fields and records into list

    for i in range(len(data_recs)):
        #dunsNumbers.append(t['results'][i]['duns'] + t['results'][i]['duns_plus4'])
        #dunsNumbers.append(t['entityData'][i]['entityRegistration']['legalBusinessName'])
        data_det = data_recs[i]['exclusionDetails']
        data_id = data_recs[i]['exclusionIdentification']
        data_dates = data_recs[i]['exclusionActions']['listOfActions'][0]
        data_addr = data_recs[i]['exclusionPrimaryAddress']
        data_txt = data_recs[i]['exclusionOtherInformation']
        
        #concatenate all dicts
        data_dict = {**data_det, **data_id, **data_dates, **data_addr,**data_txt}
        data_list.append(data_dict)
        #exit(0)
    if (len(data_recs) > 0):
        print_txt='Number of %s Exclusions found: %s/%s'%(classification,len(data_recs),search_uei)
        print_log(LOG_FILE, print_txt,1)
    else:        
        #No records found is not an issue for exclusions.
        print_txt='No %s Exclusion records found: %s/%s'%(classification,len(data_recs),search_uei)
        print_log(LOG_FILE, print_txt,1)
        return [],False
    #Convert dicts into dataframe
    df = pd.DataFrame(data_list)
    #retrieve only fields needed from config file
    df = df[data_file_cols]
    #replace SAM field names with database field names
    df.columns = data_file_fields
    
    return (df, True)

def get_usaspending_data(v_fiscal_year,data_file_fields,data_file_cols,LOG_FILE):
    #params = {"verbose" : "true"}
    headers = {'Content-Type': 'application/json'}
    #payload = {'uei':"JE73CDQUAPA7"}
    #payload={'duns':'796528263'}
    #increment = 100 if int(file_recs/10) < 100 else int(file_recs/10) 
    WORKING_DIRECTORY = os.getcwd()
    

        
    start_date = str(int(v_fiscal_year)-1)+'-10-01'
    end_date = str(v_fiscal_year)+'-09-30'
    print_txt='Fiscal dates used: start date= %s, end_date= %s'%(start_date, end_date)
    print_log(LOG_FILE, print_txt,1)
        
    payload=  {
            
            "columns": [
            "prime_award_fain",
            "prime_award_amount",
            "prime_award_base_action_date_fiscal_year",
            "prime_award_period_of_performance_start_date",
            "prime_award_period_of_performance_current_end_date",
            "prime_award_awarding_agency_name",
            "prime_awardee_duns",
            "prime_awardee_uei",
            "prime_awardee_name",
            "subaward_amount",
            "subawardee_duns",
            "subawardee_uei",
            "subawardee_name"
                ],
        
            "filters": {
                "sub_award_types": [
                    "grant"
                    ],
                "date_type": "action_date",
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date
                },
            "request_type": "award"
            #"recipient_location":["UNITED STATES"]
            #"recipient_location":{"country":["UNITED STATES"]}
          
            },
            #"location":{"country":"UNITED STATES"},
            "file_format": "csv"
        }
        
    #base = "https://api.usaspending.gov/api/v2"
    #url = "https://api.usaspending.gov/api/v2/recipient/duns/"
    base = "https://api.usaspending.gov/api/v2/"
    awards_endpoint = "bulk_download/awards/"
    url = base + awards_endpoint 
    resp = requests.post(url , json=payload, headers=headers)
            
    if resp.status_code == 200:
        print_txt='Successful request for USA Spending extract file.'
        print_log(LOG_FILE, print_txt,1)
        #print(resp.content)
    else:
        print_txt='Failed request for USA Spending extract file.\n'
        print_txt=print_txt + 'Error message: %s'%(resp.text)        
        print_log(LOG_FILE, print_txt,1)
        #print(r.status_code, r.reason)
    #r.raise_for_status()
    #r.headers
    #r.request.headers
    #print(r.content)
    data = resp.json() 
    #meta = data['page_metadata']
    file_url = data['file_url']
    status_url = data['status_url']
    file_name = data['file_name']

    #print(file_url)
    #print(status_url)
                
    # Wait for download status to be finished
    response = requests.get(status_url)
    data = response.json()
    download_status = data['status']
    print('Initial Download status is: %s'%(download_status))
    while (download_status != 'finished'):
        print('File Download not completed, please wait...')
        print('Seconds elapsed: %s'%(data['seconds_elapsed']))
        time.sleep(30)
        response = requests.get(status_url)
        data = response.json()
        download_status = data['status']
    r = requests.get(file_url)

    print_txt='Completed Download status: %s\n'%(download_status)
    print_txt=print_txt + 'Download seconds elapsed: %s'%(data['seconds_elapsed'])
    print_log(LOG_FILE, print_txt,1)

    print_txt = 'Compressed File name: %s\n'%(file_name)
    print_txt = print_txt + 'File download URL: %s'%(file_url)
    print_log(LOG_FILE, print_txt,1)

    #check if zip file?
    while not zipfile.is_zipfile(io.BytesIO(r.content)):
        print("File format is not zipped file, please wait...")
        time.sleep(10)
        r = requests.get(file_url)
            
    z = zipfile.ZipFile(io.BytesIO(r.content))
    DATA_DIR = os.path.join(WORKING_DIRECTORY, 'Data') 
    DATA_FILE_NAME = z.namelist()[0]
    #Extract into data directory
    z.extractall(DATA_DIR)
    DATA_FILE = os.path.join(DATA_DIR, DATA_FILE_NAME) 
    print_txt = 'CSV file name extracted to Data directory: %s'%(DATA_FILE_NAME)
    print_log(LOG_FILE, print_txt,1)


    #exit(0)
    #usas_data = pd.read_csv(DATA_FILE, header=0)
    #read in the data file
    try:
        data_file_data = pd.read_csv(DATA_FILE, 
            #usecols = lambda x : x.upper() in data_file_cols,
            usecols=data_file_cols,
            dtype=str,keep_default_na=False,header=0, 
            encoding='ISO-8859-1',index_col=False, low_memory=False)[data_file_cols]
    except Exception as err:
        print_txt ='Error: column name missing or changed in data file.\n'
        print_txt = print_txt+'Column name check is CASE SENSITIVE.\n'
        print_txt = print_txt+'Please update column name in data file or configuration file as needed.\n'
        print_txt = print_txt+'%s'%(err)        
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)
    #print(data_file_data.head())
    #print(data_file_data.columns)
    print_txt = 'Number of records in CSV file: %s'%(data_file_data.shape[0])
    print_log(LOG_FILE, print_txt,1)

    print_txt = 'USAS data downloaded in CSV file: %s'%(data_file_data.columns)
    print_log(LOG_FILE, print_txt,1)

    data_file_data.columns = data_file_fields
    return data_file_data

def get_fapiis_data(v_fiscal_year,data_file_fields,data_file_cols,LOG_FILE):
    #params = {"verbose" : "true"}
    
    WORKING_DIRECTORY = os.getcwd()
    headers = {'Content-Type': 'application/json',
                'Accept': 'application/json, text/plain, */*',
                'Authorization': 'Basic RkFQSUlTUEFVU0VSOnY3UkUhZzNwMipZOFV4Pw=='
                #rest of parameters not required                
                #'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
                #'Host': 'cpars.cpars.gov',
                #'Origin': 'https://www.fapiis.gov',
                #'Referer': 'https://www.fapiis.gov/'
                }
        
    #payload=  {
    data=  {
            "type": "all",
            "group": "false"
            #"file_format": "csv"
        }
        
    #base = "https://api.usaspending.gov/api/v2"
    #url = "https://api.usaspending.gov/api/v2/recipient/duns/"
    base = "https://cpars.cpars.gov/cpars/rest/"
    endpoint = "records?"
    url = base + endpoint 
    resp = requests.get(url , data, headers=headers)
            
    if resp.status_code == 200:
        print_txt='Successful request for FAPIIS extract file.'
        print_log(LOG_FILE, print_txt,1)
        #print(resp.content)
    else:
        print_txt='Failed request for FAPIIS extract file.\n'
        print_txt=print_txt + 'Error message: %s'%(resp.text)        
        print_log(LOG_FILE, print_txt,1)
        exit(1)
        #print(r.status_code, r.reason)
    DATA_DIR = os.path.join(WORKING_DIRECTORY, 'Data') 
    today = datetime.now().strftime("%m_%d_%Y_%H_%M")
    DATA_FILE_NAME = 'FAPIIS_'+today+'.csv'
    DATA_FILE = os.path.join(DATA_DIR, DATA_FILE_NAME)
    fapiis = resp.json()
    fapiis_recs = fapiis.get("records", [])
    data_file_data = pd.DataFrame(fapiis_recs)[['uei','awardeeName','recordType','recordDate','contractGrantNumber','reportingAgencyName']]
    #retrieve only fields needed from config file
    #df = df[data_file_cols]
    #replace SAM field names with database field names
    #print(data_file_data.columns)
    #print(data_file_data.head())
    #data_file_data.columns = data_file_fields
    data_file_data.columns = ['uei','awardeeName','recordType','recordDate','contractGrantNumber','reportingAgencyName']
    #df = json_normalize(data_rec)
    #today = datetime.now().strftime("%m_%d_%Y_%H_%M")
    
    data_file_data.to_csv(DATA_FILE,index=False)
    #print(df.head())
    
    #with open(DATA_FILE, "wb+") as f:
        #f.write(fapiis_recs)
    
    print_txt = 'FAPIIS CSV file is downloaded to: %s'%(DATA_FILE)
    print_log(LOG_FILE, print_txt,1)


    print_txt = 'Number of records in FAPIIS file: %s'%(data_file_data.shape[0])
    print_log(LOG_FILE, print_txt,1)
    #exit(0)
    #data_file_data.columns = data_file_fields
    return data_file_data

def get_fac_data(myAPIKEY, data_name_curr, v_fiscal_year,data_file_fields,data_file_cols,LOG_FILE):
    
    #Max record estimate 25k, set Max to 50k
    ROW_MAX = 1000000 
    #Each request set to 10k rows of data, &limit parameter
    ROW_LIMIT = 10000
    #Rows to offset after each 10k of data, &offset parameter
    ROW_OFFSET = 10000


    headers = {'Content-Type': 'application/json' #,
                #'Accept': 'application/json, text/plain, */*',
                #'Accept-Profile': 'api'  #'api_v1_0_0_beta'  #   'api' 
                #'Prefer': 'count=exact'
                }
        
    data=  {
            #"type": "all",
            #"group": "false"
            #"file_format": "csv"
        }    
    #Change upon API rollout.    
    #base = "https://api.data.gov/TEST/audit-clearinghouse/v0/dissemination/"
    base = "https://api.fac.gov/"
    endpoints = {'FAC': 'general?',
                'FAC_UEIS': 'additional_ueis?',
                'FAC_FINDINGS': 'findings?',
                'FAC_FINDINGS_TXT': 'findings_text?',
                'FAC_CFDA': 'federal_awards?'
                }
    endpoint = endpoints[data_name_curr]
    #endpoint_general = "general?" 
    apikey = "&api_key="+ myAPIKEY
    filters = "&audit_year=eq."+ str(v_fiscal_year)
    cfda_filter = "&federal_agency_prefix=eq.94"
    if data_name_curr == 'FAC_CFDA':
        filters = filters+cfda_filter

    #Use &limit and &offset to download in chunks of data, otherwise fails around 18k limit
    df_data_name = pd.DataFrame([])
    for offset in range(0,ROW_MAX,ROW_OFFSET):
        limit = "&limit="+ str(ROW_LIMIT) + "&offset=" + str(offset)
        url_data_name = base + endpoint + apikey + filters + limit
        #url_general = base + endpoint_general + apikey + filters_general + limit
        #print("Query url for %s is: %s"%(data_name_curr,url_data_name))
    
        resp = requests.get(url_data_name,headers=headers)
        #resp = requests.get(url_general)
        
        if resp.status_code == 200:
            print_txt='Successful request for %s data, for audit year=%s, current offset=%s'%(data_name_curr,str(v_fiscal_year),str(offset))
            print_log(LOG_FILE, print_txt,1)
            #print(resp.content)
            
        else:
            print_txt='Failed request for %s data, for audit year=%s, current offset=%s.\n'%(data_name_curr,str(v_fiscal_year),str(offset)) 
            print_txt=print_txt + 'Error message: %s'%(resp.text)        
            print_log(LOG_FILE, print_txt,1)
            exit(1)
        #print(r.status_code, r.reason)
        fac_data_name = resp.json()
        #current chunk of data
        
        df_data_name_limit = pd.DataFrame(fac_data_name)
        #print(df_data_name_limit.head())
        #exit(0)
        #if offset == 0:
            #df_general = df_general_limit
        df_data_name = pd.concat([df_data_name, df_data_name_limit],axis=0)
        #break if all records retrieved
        if (df_data_name_limit.shape[0] < ROW_OFFSET):
            break         
    
    num_records_data_name = df_data_name.shape[0]
    if num_records_data_name == 0:
        print_txt='No data found for %s data, for audit year=%s'%(data_name_curr,str(v_fiscal_year))
        print_log(LOG_FILE, print_txt,1)
        data_file_data = pd.DataFrame([])
        return data_file_data  
            

  
    #selet only potential useful columns
    data_file_data = df_data_name[data_file_cols]
    data_file_data.columns = data_file_fields
    
    #replace boolean with Y/N
    replace_txt = {'Yes':'Y','No':'N'}
    data_file_data = data_file_data.replace(replace_txt,regex=False)
    replace_txt = {'unmodified_opinion':'U',
                   'qualified_opinion':'Q',
                   'adverse_opinion': 'A',
                   'disclaimer':'D',
                   'not_gaap': 'S'}
    data_file_data = data_file_data.replace(replace_txt,regex=True)
    #Clean up data to remove list like columns and limit size to 256 chars
    #data_file_data['MULTIPLEEINS'] = ['-'.join(map(str, eins))[:256] for eins in data_file_data['MULTIPLEEINS']]

    if data_name_curr == 'FAC':
        #change date format to dd-mon-yy so oracle auto converts to date type
        data_file_data['FACACCEPTEDDATE'] = [ pd.to_datetime(fac_date).strftime('%d-%b-%y') for fac_date in data_file_data['FACACCEPTEDDATE']]
        data_file_data['FYENDDATE'] = [ pd.to_datetime(fac_date).strftime('%d-%b-%y') for fac_date in data_file_data['FYENDDATE']]
        data_file_data['AUDITEEDATESIGNED'] = [ pd.to_datetime(fac_date).strftime('%d-%b-%y') for fac_date in data_file_data['AUDITEEDATESIGNED']]
        data_file_data['CPADATESIGNED'] = [ pd.to_datetime(fac_date).strftime('%d-%b-%y') for fac_date in data_file_data['CPADATESIGNED']]
        data_file_data['DATEFIREWALL'] = [ pd.to_datetime(fac_date).strftime('%d-%b-%y') for fac_date in data_file_data['DATEFIREWALL']]
        #Left pad EIN with leading zeros to 9 chars
        data_file_data['EIN'] = [ str(x).zfill(9) for x in data_file_data['EIN']]  
        #Concatenate comma delimited Typereport_fs
        data_file_data['TYPEREPORT_FS'] = [ ''.join(str(x).split(',')).replace(' ', '') for x in data_file_data['TYPEREPORT_FS']]  
    #print(data_file_data.head())
    #data_file_data = data_file_data.drop(['MULTIPLEDUNS'], axis=1)
    #Collapse multiple rows findings text into one
    elif data_name_curr == 'FAC_FINDINGS_TXT':
        data_file_data = data_file_data.groupby(['REPORTID','AUDITYEAR','AUDITEE_UEI','FINDINGREFNUMS','CHARTSTABLES'])['TEXT'].apply(' '.join).reset_index() 
        data_file_data = data_file_data[['REPORTID','AUDITYEAR','AUDITEE_UEI','FINDINGREFNUMS','TEXT','CHARTSTABLES']]
        data_file_data.columns = data_file_fields   
        #print(data_file_data.head())
        #exit(0)
    elif data_name_curr == 'FAC_CFDA':
        cfda_codes= ['94']            
        data_file_data = data_file_data[data_file_data['CFDA'].isin(cfda_codes)] #isin(cfda_codes)]
        if data_file_data.shape[0] == 0:
            print_txt='No CFDA=94 data found for %s, for audit year=%s'%(data_name_curr,str(v_fiscal_year))
            print_log(LOG_FILE, print_txt,1)
            return data_file_data  
 

    return data_file_data

def get_fac_rollups(data_file_data,data_file_fields,LOG_FILE):
    rollup_fields = ['COG_OVER','CYFINDINGS','PYSCHEDULE','QCOSTS','MATERIALWEAKNESS_MP','REPORTABLECONDITION_MP','TYPEREPORT_MP']
    add_fields = ['COG_OVER']
    data_file_data['FAC']['COG_OVER'] = [ 'C' if x else 'O' for x in data_file_data['FAC']['COGAGENCY']]
    df_findings = data_file_data['FAC_FINDINGS']
    findings_list = list(df_findings['REPORTID'])
    df_cfda = data_file_data['FAC_CFDA']
    print_txt = 'Begin processing rollup fields %s'%(str(rollup_fields))
    print_log(LOG_FILE, print_txt,1)
    #data_file_data['FAC']['CYFINDINGS'] = [ 'Y' if findings_exist(year, reportid) else 'N' for (year, reportid) in data_file_data['FAC'].loc[:,['AUDITYEAR','REPORTID']]]
    #data_file_data['FAC']['CYFINDINGS'] = [ 'Y' if data_file_data['FAC_FINDINGS'].apply(lambda x: findings_exist(x,str(year+reportid)),axis=1).any() else 'N' for (year, reportid) in zip(data_file_data['FAC']['AUDITYEAR'],data_file_data['FAC']['REPORTID'])]
    #Performance issues with apply lamda function, replace with list lookup below
    #data_file_data['FAC']['CYFINDINGS'] = [ 'Y' if df_findings.apply(lambda x: CYfindings(x,reportid),axis=1).any() else 'N' for reportid in data_file_data['FAC']['REPORTID']]
    data_file_data['FAC']['CYFINDINGS'] = [ 'Y' if reportid in findings_list else 'N' for reportid in data_file_data['FAC']['REPORTID']]
    data_file_data['FAC']['PYSCHEDULE'] = [ 'N' if x=="00" else 'Y' for x in data_file_data['FAC']['AGENCIES_WITH_PRIOR_FINDINGS']]
    data_file_data['FAC']['QCOSTS'] = [ 'Y' if Qcosts(df_findings,reportid) else 'N' for reportid in data_file_data['FAC']['REPORTID']]
    data_file_data['FAC']['MATERIALWEAKNESS_MP'] = [ 'Y' if Materialweakness_MP(df_findings,reportid) else 'N' for reportid in data_file_data['FAC']['REPORTID']]
    data_file_data['FAC']['REPORTABLECONDITION_MP'] = [ 'Y' if Reportablecondition_MP(df_findings,reportid) else 'N' for reportid in data_file_data['FAC']['REPORTID']]
    data_file_data['FAC']['TYPEREPORT_MP'] = [ Typereport_MP(df_cfda,reportid) for reportid in data_file_data['FAC']['REPORTID']]
    data_file_fields['FAC'].extend(add_fields)
    print_txt = 'Successfully processed rollup fields %s'%(str(rollup_fields))
    print_log(LOG_FILE, print_txt,1)

    return (data_file_data['FAC'], data_file_fields['FAC'])

def CYfindings(x, reportid):
    
    #print('Series is %s'%str(x))
    #print('Year and report id are %s'%year_reportid)
    if reportid in list(x):
        return True              
    return False
def Qcosts(df_findings, reportid):

    replace_txt = {'Y':1,'N':0}
    df_reportid = df_findings[df_findings['REPORTID']==reportid]
    if not df_reportid.empty:
        if df_reportid['QCOSTS'].replace(replace_txt,regex=False).sum() > 0:
            return True              
    return False

def Materialweakness_MP(df_findings, reportid):

    replace_txt = {'Y':1,'N':0}
    df_reportid = df_findings[df_findings['REPORTID']==reportid]
    if not df_reportid.empty:
        if df_reportid['MATERIALWEAKNESS'].replace(replace_txt,regex=False).sum() > 0:
            return True              
    return False
def Reportablecondition_MP(df_findings, reportid):

    replace_txt = {'Y':1,'N':0}
    df_reportid = df_findings[df_findings['REPORTID']==reportid]
    if not df_reportid.empty:
        if df_reportid['SIGNIFICANTDEFICIENCY'].replace(replace_txt,regex=False).sum() > 0:
            return True              
    return False    

def Typereport_MP(df_cfda, reportid):

    df_reportid = df_cfda[df_cfda['REPORTID']==reportid]
    if not df_reportid.empty:
        #typereport = df_reportid['TYPEREPORT_MP'].apply(''.join)
        typereport = df_reportid['TYPEREPORT_MP'].drop_duplicates(keep='first')
        str_typereport = ''.join(list(typereport))        
        if str(str_typereport) == 'U':
            unique_typereport = str_typereport
        else:
            unique_typereport = str_typereport.replace('U','')
        #if reportid == '2023-06-GSAFAC-0000004925':
            #print('report id = %s, string=%s and typereport = %s'%(reportid,str_typereport,unique_typereport))
            #exit(0)

        return unique_typereport
    
    return     

def extract_cncs_recs(cncs_fac_reportids,data_name_curr, data_file_data, LOG_FILE):    
    # 

    data_file_data_cncs = pd.merge(cncs_fac_reportids,
                 data_file_data[data_name_curr],
                 left_on='REPORTID_FAC', right_on='REPORTID', how='inner')
    
    data_file_data_cncs = data_file_data_cncs.drop(['REPORTID_FAC'], axis=1)
    #exit(1)
    print_txt = 'Americorps orgs audit recs found for %s: %s'%(data_name_curr,data_file_data_cncs.shape[0])
    print_log(LOG_FILE, print_txt,1)

    return data_file_data_cncs

def get_cncs_reportids(cursor,data_file_data_fac,LOG_FILE):
    sql = """select distinct uei_nbr, case when uei_nbr is null then ein else null end EIN from organizations o 
            """

       #cngrants_duns = pd.read_sql(sql, con=connection)   
    cursor.execute(sql)
 
    cngrants_orgs = pd.DataFrame(cursor.fetchall(),columns=['UEI_NBR','EIN_NBR'])
    #cngrants_uei = cngrants[cngrants['UEI_NBR'].notnull()]
        
    print_txt = 'Number of Orgs in Organizations table: %s'%cursor.rowcount
    print_log(LOG_FILE, print_txt,1)

    cngrants_fac_uei = pd.merge(cngrants_orgs,
                 data_file_data_fac,
                 left_on='UEI_NBR', right_on='UEI', how='inner')
    cngrants_fac_ein = pd.merge(cngrants_orgs,
                 data_file_data_fac,
                 left_on='EIN_NBR', right_on='EIN', how='inner')
    
    cngrants_fac = pd.concat([cngrants_fac_uei, cngrants_fac_ein],axis=0)
    
    cngrants_fac = cngrants_fac.drop(['UEI_NBR','EIN_NBR'], axis=1)
    print_txt = 'Number of AmeriCorps only FAC records: %s'%cngrants_fac.shape[0]
    print_log(LOG_FILE, print_txt,1)
    return cngrants_fac[['REPORTID']]

def print_to_csv(v_fiscal_year,data_name_curr,data_file_data,LOG_FILE):    
    #Print to file 
    WORKING_DIRECTORY = os.getcwd()
    DATA_DIR = os.path.join(WORKING_DIRECTORY, 'Data') 
    today = datetime.now().strftime("%m_%d_%Y_%H_%M")
    DATA_FILE_NAME = data_name_curr+'_'+str(v_fiscal_year)+'_'+today+'.csv'
    DATA_FILE = os.path.join(DATA_DIR, DATA_FILE_NAME)
    data_file_data[data_name_curr].to_csv(DATA_FILE,index=False)
    #exit(1)
    print_txt = '%s CSV file is saved to: %s'%(data_name_curr,DATA_FILE)
    print_log(LOG_FILE, print_txt,1)

    print_txt = '%s CSV file number of records is: %s'%(data_name_curr,data_file_data[data_name_curr].shape[0])
    print_log(LOG_FILE, print_txt,1)

    return    




def get_fapiis_sam_data(myAPIKey,cursor,data_file_fields,data_file_cols,LOG_FILE):


    #3baseURL = "https://api.sam.gov/entity-information/v2/entities?"
    #baseURL = baseURL + "&api_key=" + myAPIKey
    baseURL = "https://api.sam.gov/entity-information/v3/entities?"
    baseURL = baseURL + "&api_key=" + myAPIKey + "&includeSections=integrityInformation"

    #Collect organizations in Monitoring/Applications universe
        #sql = """select distinct uei_nbr,name_upper ORG_NAME
        #    from ares.mv_daily_app_universe mu, organizations o where mu.org_id = o.id """
    sql = """select distinct uei_nbr
            from ares.mv_daily_app_universe mu, organizations o where mu.org_id = o.id 
            and uei_nbr is not null
            """

       #cngrants_duns = pd.read_sql(sql, con=connection)   
    cursor.execute(sql)
    #cngrants = pd.DataFrame(cursor.fetchall(),columns=['UEI_NBR','ORG_NAME'])
    cngrants = pd.DataFrame(cursor.fetchall(),columns=['UEI_NBR'])
    #cngrants_uei = cngrants[cngrants['UEI_NBR'].notnull()]
        
    print_txt = 'Number of Orgs in Application Universe: %s'%cursor.rowcount
    print_log(LOG_FILE, print_txt,1)
    #cngrants_duns.columns = [x[0] for x in cursor.description]
 
    #limit to orgs in CN_GRANTS using non null UEI
    cngrants_fapiis = pd.DataFrame([], columns = data_file_fields)
    search_key = ['UEI','ONAME']
    uei_set = []
    oname_set = []
    
    #print(range(len(cngrants_uei['UEI_NBR'])))
    for i in range(len(cngrants['UEI_NBR'])):
        uei_set.append(cngrants['UEI_NBR'][i])
        if (((i+1)%50 == 0)  or (i==(len(cngrants['UEI_NBR'])-1))):
            #Limit to 100 uei per transaction, less than 2k url limit
            uei_str = '~'.join(uei_set)
            fapiis_recs, rec_found = get_fapiis_sam_recs(baseURL,data_file_cols, data_file_fields, uei_str, search_key[0], LOG_FILE)
            if rec_found:
                cngrants_fapiis = pd.concat([cngrants_fapiis, fapiis_recs],axis=0)
            uei_set = []

    data_file_data = cngrants_fapiis
    WORKING_DIRECTORY = os.getcwd()
    DATA_DIR = os.path.join(WORKING_DIRECTORY, 'Data') 
    today = datetime.now().strftime("%m_%d_%Y_%H_%M")
    DATA_FILE_NAME = 'FAPIIS_SAM_'+today+'.csv'
    DATA_FILE = os.path.join(DATA_DIR, DATA_FILE_NAME)
    data_file_data.to_csv(DATA_FILE,index=False)
        
    print_txt ='Fapiis from SAM Entities API records to be loaded: %s'%(len(data_file_data.index))        
    print_log(LOG_FILE, print_txt,1)
    return data_file_data

def get_fapiis_sam_recs(baseURL,data_file_cols, data_file_fields, key_val, search_key, LOG_FILE):
    
    if (search_key == 'UEI'):
        parameter = "&ueiSAM=["+key_val+"]"
    else: 
        key_val = key_val.replace('&', '')
        parameter = "&legalBusinessName=[%22"+key_val+"%22]"
        
    #queryURL = baseURL + qterms + "&api_key=" + myAPIKey
    queryURL = baseURL + parameter
    search_uei = len(key_val.split('~'))
    #print(queryURL)

    #else:
        #print('Successfully retreived data from ' + queryURL)
    data_recs = []
    data_list = []
    data_dict = {}
    new_results = True
    page = 0
    while new_results:
        try:
            fapiis = requests.get(queryURL + f"&page={page}&size=10").json()
        except Exception as e:
                print_txt='Request for FAPIIS data failed: %s\n'%queryURL
                print_txt=print_txt + 'Exception message is: %s'%str(e)
                print_log(LOG_FILE, print_txt,2)                                
                exit(1)

        new_results = fapiis.get("entityData", [])
        data_recs.extend(new_results)
        page += 1
    
    #extract fields and records into list
    for i in range(len(data_recs)):

        data_sum = data_recs[i]['integrityInformation']['entitySummary']
        #print(f'UEI is {data_sum["ueiSAM"]}')
        fapiis_count = data_recs[i]['integrityInformation']['responsibilityInformationCount']
        for j in range(fapiis_count):                       
            #add try exception block here
            try:
                data_fap = data_recs[i]['integrityInformation']['responsibilityInformationList'][j]            
            except Exception as e:
                print_txt='Error getting FAPIIS data for %s\n'%data_sum['ueiSAM','legalBusinessName']
                print_txt=print_txt + 'Exception message is: %s'%str(e)
                print_log(LOG_FILE, print_txt,2)                                
                continue

            data_dict = {**data_sum, **data_fap}
            data_list.append(data_dict)
        
    #exit(0)   
    if (len(data_list) > 0):
        for i in range(len(data_list)):
            print_txt='Fapiis rec found for: %s'%(data_list[i]["legalBusinessName"])
            print_log(LOG_FILE, print_txt,1)
    
    else:    
        #print_txt='No FAPIIS records found for %s Entities.'%(search_uei)  
        #print(print_txt)
        #with open(LOG_FILE, 'a') as f:
            #f.write(print_txt)
            #f.write('\n')
        return [],False
    #Convert list of dicts into dataframe
    df = pd.DataFrame(data_list)
    #retrieve only fields needed from config file
    df = df[data_file_cols]
    #replace SAM field names with database field names
    df.columns = data_file_fields
    
    return (df, True)

def connect_oracle(v_db_env, LOG_FILE):
    #print(data_file_data.head())
    #data_file_data.to_csv(OUTPUT_FILE, index=False)
     #print("ARCH:", platform.architecture())
    lib_dir = r"C:\oracle\Product\instantclient_21_3"
    #Turn off warning SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    
    try:
        cx_Oracle.init_oracle_client(lib_dir=lib_dir)
    except Exception as err:
        print_txt = "Error connecting: cx_Oracle.init_oracle_client()\n"
        print_txt = print_txt+err
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)


    CURRENT_DIRECTORY = os.getcwd()    
    #ctx_auth = AuthenticationContext(SITE_URL)
    PSWD_FILE = os.path.join(CURRENT_DIRECTORY, 'Config','pswd.txt') 
    if not os.path.exists(PSWD_FILE):
        print_txt='Error: Credentials file was not found: %s '%(PSWD_FILE)
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)
    with open(PSWD_FILE) as file:
        lines = file.readlines()
    for line in lines:
        env=line.rstrip().split('=')[0]
        if (env.upper() == v_db_env.upper()):
            user_id=line.rstrip().split('=')[1].split(',')[0]
            passwd=line.rstrip().split('=')[1].split(',')[1]        
    #print('User name is: %s, password is: %s'%(user_id, passwd))


    #user_passwd = v_credentials.split('/')
    #user_id = user_passwd[0]
    #passwd = user_passwd[1]
    #print('User: '+ user + ' passwd: '+ passwd)
    #return
    #print(cx_Oracle.clientversion())
    db_dsn = 'DSN NAME REMOVED' 
    try:
        connection = cx_Oracle.connect(user=user_id, password=passwd,
                               dsn=db_dsn)
    except Exception as err:
        print_txt='Error connecting: Invalid User/Password. Please check and try again.'
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)

    cursor = connection.cursor()    
    return (connection, cursor)
    
    
def load_data(connection, cursor, ra_table,data_name_curr, data_file_headers, data_file_data, LOG_FILE):
    
    start = time.time()
    print_txt='Data import start time: %s'%datetime.now().strftime("%m_%d_%Y %H:%M:%S")
    print_log(LOG_FILE, print_txt,2)
    #Limit USA Spending to only CNCS Organizations
    if data_name_curr == 'USAS':
        #Collect organizations in Monitoring/Applications universe
        sql = """select distinct o.duns_nbr from ares.mv_daily_mon_universe mu, organizations o where mu.org_id = o.id
               and o.duns_nbr is not null
               union
             select distinct o.duns_nbr from ares.mv_daily_app_universe mu, organizations o where mu.org_id = o.id
               and o.duns_nbr is not null      """

        #cngrants_duns = pd.read_sql(sql, con=connection)   
        cursor.execute(sql)
        
        cngrants_duns = pd.DataFrame(cursor.fetchall(),columns=['DUNS_NBR'])
        print_txt = 'DUNS Numbers in Monitoring/Application Universe: %s'%cursor.rowcount
        print_log(LOG_FILE, print_txt,1)
        #cngrants_duns.columns = [x[0] for x in cursor.description]
        #Collect orgs already loaded
        sql_loaded = """select prime_awardee_duns , 
                    prime_award_fiscal_year,subawardee_duns,
                    prime_award_awarding_agency_nm,prime_award_fain  from """+ra_table+""" 
                    group by prime_awardee_duns, prime_award_fiscal_year,subawardee_duns,
                    prime_award_awarding_agency_nm,prime_award_fain """
        cursor.execute(sql_loaded)        
        ra_usas_duns_fy = pd.DataFrame(cursor.fetchall(), columns=['PRIME_AWARDEE_DUNS_TBL','PRIME_AWARD_FISCAL_YEAR_TBL',
                                'SUBAWARDEE_DUNS_TBL','PRIME_AWARD_AWARDING_AGENCY_NM_TBL',
                                'PRIME_AWARD_FAIN_TBL'])
        print_txt='Prime DUNS-Fiscal Year-Sub DUNS combination previously loaded: %s'%cursor.rowcount
        print_log(LOG_FILE, print_txt,1)
        #print(cursor.description)
        #return
        #ra_usas_duns_fy.columns = [x[0] for x in cursor.description]


        fiscal_year = date.today().year+1 if date.today().month>9 else date.today().year

        #limit to orgs in CN_GRANTS only Part 1
        cngrants_usas = pd.merge(cngrants_duns,
                 data_file_data,
                 left_on='DUNS_NBR', right_on='PRIME_AWARDEE_DUNS', how='inner')
        data_file_data = cngrants_usas.drop('DUNS_NBR', axis=1)
        #print(data_file_data.columns)
        #filter data already loaded Part 2
        usas_new = pd.merge(data_file_data,ra_usas_duns_fy,
                 left_on=['PRIME_AWARDEE_DUNS','PRIME_AWARD_FISCAL_YEAR','SUBAWARDEE_DUNS','PRIME_AWARD_AWARDING_AGENCY_NM',
                                'PRIME_AWARD_FAIN'],
                 right_on=['PRIME_AWARDEE_DUNS_TBL', 'PRIME_AWARD_FISCAL_YEAR_TBL','SUBAWARDEE_DUNS_TBL','PRIME_AWARD_AWARDING_AGENCY_NM_TBL',
                                'PRIME_AWARD_FAIN_TBL'], how='left')
        usas_new = usas_new[usas_new['PRIME_AWARDEE_DUNS_TBL'].isnull()]
        data_file_data = usas_new.drop(['PRIME_AWARDEE_DUNS_TBL', 'PRIME_AWARD_FISCAL_YEAR_TBL','SUBAWARDEE_DUNS_TBL',
                        'PRIME_AWARD_AWARDING_AGENCY_NM_TBL','PRIME_AWARD_FAIN_TBL'], axis=1)
        #3-15 Remove duplicate subawardees at grant level using max subaward amount
        data_file_data.sort_values(['PRIME_AWARDEE_DUNS','PRIME_AWARD_FISCAL_YEAR',
                                    'SUBAWARDEE_DUNS','PRIME_AWARD_AWARDING_AGENCY_NM',
                                'PRIME_AWARD_FAIN','SUBAWARD_AMOUNT'],ascending=False,inplace=True)
        #3-15 Remove duplicate subawardees, pick highest subaward amount for each grant
        data_file_data.drop_duplicates(['PRIME_AWARDEE_DUNS','PRIME_AWARD_FISCAL_YEAR',
                                    'SUBAWARDEE_DUNS','PRIME_AWARD_AWARDING_AGENCY_NM',
                                'PRIME_AWARD_FAIN'],keep='first',inplace=True)
        
        print_txt ='New Prime DUNS-Fiscal Year-Sub DUNS combination to be loaded: %s'%(len(data_file_data.index))        
        print_log(LOG_FILE, print_txt,1)
        #print(data_file_data[['PRIME_AWARDEE_DUNS','PRIME_AWARD_FISCAL_YEAR',
        #                            'SUBAWARDEE_DUNS','SUBAWARD_AMOUNT']].head(10))
        #return
        #keep only last 3 fiscal years
        data_file_data = data_file_data[data_file_data['PRIME_AWARD_FISCAL_YEAR']>str(fiscal_year-4)]
        print_txt = 'New Prime DUNS-Fiscal Year-Sub DUNS for last 3 fiscal years: %s'%(len(data_file_data.index))
        print_log(LOG_FILE, print_txt,1)
        #data_file_data = usas_new

    #print(data_file_data.columns)
    #print(len(data_file_data.index))
    #print ("Data current is %s and headers %s"%(data_name_curr, str(data_file_headers)))
    col_name = data_file_headers
    #exit(0)
    col_name_str = """ ("""+','.join(col_name)+""") """
    #vals = ['23456','Test Organization Name']
    #f = lambda x: "N/A" if pd.isna(x) else ("" if pd.isnull(x) else str(x))
    g = lambda x: "" if pd.isnull(x) else str(x)
    vals_vars = """ (:"""+',:'.join(col_name)+""")"""
    #sql = """ insert into """+ra_table+""" (application_id) values( :did)"""
    sql = """ insert into """+ra_table+ col_name_str + """values""" + vals_vars   

    rowcount = 0
    file_recs = len(data_file_data.index)
    increment = 100 if int(file_recs/10) < 100 else int(file_recs/10) 
    increment = int(increment/100)*100
    for index, rows in data_file_data.iterrows():       
        vals = [g(x) for x in rows]                
        try:
            cursor.execute(sql,vals)
        except Exception as e:
            error, = e.args
            if error.code in (12899,1438):
                print_txt = error.message
                print_log(LOG_FILE, print_txt,1)

                continue
            else:
                print_txt = error.message
                print_log(LOG_FILE, print_txt,1)
                exit(1)
        rowcount += 1
        if (rowcount%increment == 0):
            print('Inserted records %s/%s ...'%(rowcount,file_recs))
    connection.commit()
    print_txt='Successfully imported records into %s: %s/%s'%(ra_table,rowcount,file_recs)
    print_log(LOG_FILE, print_txt,1)
        
    #Update create date with sysdate    
    sql = """ UPDATE """+ra_table+ """ SET CREATE_DT = SYSDATE
                WHERE CREATE_DT IS NULL"""  
    cursor.execute(sql)
    connection.commit()
    runtime = (time.time() - start)/60
    print_txt='Data import completion: %s \n'%datetime.now().strftime("%m_%d_%Y %H:%M:%S")
    print_runtime='Import duration: %d mins'%runtime
    print_txt = print_txt+print_runtime
    print_log(LOG_FILE, print_txt,2)
        
def print_log(LOG_FILE, print_txt, target): 
    if target == 1:
        print(print_txt) 
    with open(LOG_FILE, 'a') as f:
        f.write(print_txt)
        f.write('\n')


def main( v_db_env, v_data_name, v_fiscal_year):
#def main():
    #examp

    DEMO_MAX = 20000
    num_rows = 0
    VERSION_CODE = 24
    VERSION_CONFIG = 21

    WORKING_DIRECTORY = os.getcwd()
    #print('Imported Index directory is '+ INDEX_DIRECTORY)
    #print('Imported CACHE directory is '+ IRSX_CACHE_DIRECTORY)
    #print('Imported XML directory is '+ WORKING_DIRECTORY)
    today = datetime.now().strftime("%m_%d_%Y_%H_%M")
    tables_map = {
                    'USAS': 'RA_USASPENDING',                  
                    'SAM_REG':'RA_SAM_REGISTRATIONS',
                    'SAM_EXCL':'RA_SAM_EXCLUSIONS',
                    'FAPIIS': 'RA_FAPIIS',
                    'FAC': 'RA_FAC_API',
                    'FAC_UEIS': 'RA_FAC_UEIS_API',
                    'FAC_FINDINGS': 'RA_FAC_FINDINGS_API',
                    'FAC_FINDINGS_TXT':'RA_FAC_FINDINGS_TXT_API',
                    'FAC_CFDA':'RA_FAC_CFDA_API',
                    }
    #ra_table = 'ARES.'+tables_map[v_data_name]    
    #ra_table = 'DAMARE.'+tables_map[data_name_curr] 
    
    #If fiscal year not provided, set to current fiscal year
    if (v_fiscal_year == None):
        #apply current fiscal year
        v_fiscal_year = int(date.today().year)+1 if date.today().month > 9  else date.today().year
    #Add processed year to log files if USAS, FAC
    if v_data_name in ['USAS','FAC']:
        data_file_log = v_data_name + '_'+ str(v_fiscal_year)+'_log_'+today+'.txt'
    else:
        data_file_log = v_data_name + '_log_'+today+'.txt'

    CONFIG_FILE = os.path.join(WORKING_DIRECTORY, 'Config','Data_API_structs_v'+str(VERSION_CONFIG)+'.xls') 
    API_KEY_FILE = os.path.join(WORKING_DIRECTORY, 'Config','api_key.txt') 
    LOG_FILE = os.path.join(WORKING_DIRECTORY, 'Logs',data_file_log) 
    
    print_txt='Data load script version: %s\n'%(VERSION_CODE)
    print_txt=print_txt + 'Config file version: %s'%(VERSION_CONFIG)    
    print_log(LOG_FILE, print_txt,1)
    
    #print(myAPIKey)    
    #exit(0)
    if not os.path.exists(CONFIG_FILE):
        print_txt='Error: Config file was not found: %s '%(CONFIG_FILE)
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)
    else:
        print_txt = "Config file: %s" %(CONFIG_FILE) 
        print_log(LOG_FILE, print_txt,1)
    
    connection, cursor = connect_oracle(v_db_env, LOG_FILE)
    sam_files = ['SAM_REG','SAM_EXCL']
    if v_data_name in ['SAM','FAPIIS','FAC']:
        #Locate the API KEY
        if not os.path.exists(API_KEY_FILE):
            print_txt='Error: API KEY file was not found: %s '%(API_KEY_FILE)
            print_log(LOG_FILE, print_txt,1)
            sys.exit(1)
        with open(API_KEY_FILE) as file:
            #line = file.readlines()
            #myAPIKey = line[0].rstrip().split('=')[1]
            for line in file:
                if v_data_name in ['SAM','FAPIIS'] and line.rstrip().split('=')[0].lower() == 'api_key_sam':
                    myAPIKEY = line.rstrip().split('=')[1] 
                elif v_data_name in ['FAC'] and line.rstrip().split('=')[0].lower() == 'api_key_fac':
                    myAPIKEY = line.rstrip().split('=')[1] 
    

    #print("API key is: %s "%(myAPIKEY))                
    #exit(0)

    if v_data_name == 'SAM':
        #Load Registrations and Exclusions
        for data_name_curr in sam_files:
            #Extract column and field headers from config file
            data_file_headers = pd.read_excel(CONFIG_FILE, names=['FIELDS','COLUMNS'],sheet_name = data_name_curr,header=None, index_col=False)
            data_file_fields = data_file_headers['FIELDS'].T.tolist()
            data_file_cols = data_file_headers['COLUMNS'].T.tolist()
            #table name to load
            ra_table = 'ARES.'+tables_map[data_name_curr]    
            #Retrieve all registration records from SAM.gov
            if data_name_curr == 'SAM_REG':
                data_file_data = get_registrations_data(myAPIKEY,connection, cursor,ra_table,data_file_fields,data_file_cols,LOG_FILE)
            #Retrieve all exclusion records from SAM.gov            
            if data_name_curr == 'SAM_EXCL':
                data_file_data = get_exclusions_data(myAPIKEY,connection, cursor,ra_table,data_file_fields,data_file_cols,LOG_FILE)            
            #Load records into Oracle
            load_data(connection, cursor,ra_table,data_name_curr, data_file_fields, data_file_data, LOG_FILE)
    
    if v_data_name == 'USAS':
        data_name_curr = v_data_name
        data_file_headers = pd.read_excel(CONFIG_FILE, names=['FIELDS','COLUMNS'],sheet_name = data_name_curr,header=None, index_col=False)
        data_file_fields = data_file_headers['FIELDS'].T.tolist()
        data_file_cols = data_file_headers['COLUMNS'].T.tolist()
        ra_table = 'ARES.'+tables_map['USAS'] 
        data_file_data = get_usaspending_data(v_fiscal_year,data_file_fields,data_file_cols,LOG_FILE)        
        load_data(connection, cursor,ra_table,data_name_curr, data_file_fields, data_file_data, LOG_FILE)

    if v_data_name == 'FAPIIS':
        data_name_curr = v_data_name
        data_file_headers = pd.read_excel(CONFIG_FILE, names=['FIELDS','COLUMNS'],sheet_name = data_name_curr,header=None, index_col=False)
        data_file_fields = data_file_headers['FIELDS'].T.tolist()
        data_file_cols = data_file_headers['COLUMNS'].T.tolist()
        ra_table = 'ARES.'+tables_map['FAPIIS'] 
        data_file_data = get_fapiis_sam_data(myAPIKEY,cursor,data_file_fields,data_file_cols,LOG_FILE)
        load_data(connection, cursor,ra_table,data_name_curr, data_file_fields, data_file_data, LOG_FILE)
    if v_data_name == 'FAC':
        #data_name_curr = v_data_name
        fac_files = ['FAC','FAC_UEIS','FAC_FINDINGS','FAC_FINDINGS_TXT','FAC_CFDA'] 
        #fac_files = ['FAC']
        data_file_data = dict.fromkeys(fac_files) 
        data_file_fields = dict.fromkeys(fac_files)
        data_file_cols = dict.fromkeys(fac_files)
        #fac_files = ['FAC_FINDINGS_TXT']
        for data_name_curr in fac_files:
            data_file_headers = pd.read_excel(CONFIG_FILE, names=['FIELDS','COLUMNS'],sheet_name = data_name_curr,header=None, index_col=False)
            #data_file_fields = data_file_headers['FIELDS'].T.tolist()
            data_file_fields[data_name_curr] = data_file_headers['FIELDS'].T.tolist()
            #data_file_cols = data_file_headers['COLUMNS'].T.tolist()
            data_file_cols[data_name_curr] = data_file_headers['COLUMNS'].T.tolist()            
            data_file_data[data_name_curr] = get_fac_data(myAPIKEY, data_name_curr, v_fiscal_year,data_file_fields[data_name_curr],data_file_cols[data_name_curr],LOG_FILE)  
        #Limit data to CNCS uei and ein only, returns cncs fac reportids.
        #print('Before cncs extract fac rec numbers %s'(data_file_data['FAC'].shape[0]))
        cncs_fac_reportids = get_cncs_reportids(cursor,data_file_data['FAC'],LOG_FILE)
        cncs_fac_reportids.columns = ['REPORTID_FAC']

        #Limit all datasets to CNCS orgs only
        for data_name_curr in fac_files:    
            if not data_file_data[data_name_curr].empty:
                data_file_data[data_name_curr] = extract_cncs_recs(cncs_fac_reportids,data_name_curr, data_file_data, LOG_FILE)
            
        #Get rollup data from cross-walks before importing data    
        for data_name_curr in fac_files:    
            ra_table = 'ARES.'+tables_map[data_name_curr]                         
            if not data_file_data[data_name_curr].empty:
                if data_name_curr == 'FAC':
                    data_file_data[data_name_curr], data_file_fields[data_name_curr] = get_fac_rollups(data_file_data,data_file_fields,LOG_FILE)
                print_to_csv(v_fiscal_year,data_name_curr,data_file_data, LOG_FILE)
                load_data(connection, cursor,ra_table,data_name_curr, data_file_fields[data_name_curr], data_file_data[data_name_curr], LOG_FILE)

    #print to log file
    sys.exit(0)


if __name__ == '__main__':
    
    if len(sys.argv) < 3:        
        #raise ValueError('FAILED -- Invalid job execution format.')
        print('Error: Invalid job execution format.\n'+
                    'Please using following format to execute the data load script.\n'+
                            'Data_Load_API.py [Environment] [Data file type].\n'+
                            'Example: Data_Load_API.py Test SAM')
        sys.exit(1)
    v_db_env = sys.argv[1]
    #v_credentials = sys.argv[2]
    v_data_name = sys.argv[2]
    
    #v_data_file = sys.argv[4]
    if (v_db_env.lower() not in ['test','prod']):
        print('Error: Invalid environment.\n'+
                    'Please use either Test/Prod for environment.\n'+ 
                            'Data_Load_API.py [Environment] [Data file type].\n'+
                            'Example: Data_Load_API.py Test SAM')
        sys.exit(1)
        
    data_files = ['USAS','SAM','FAPIIS','FAPIIS_SAM','FAC']
    if (v_data_name not in data_files):
        print('Error: Invalid data file format.\n'+
                    'Please use one of following data file types.\n'+
                        'SAM|USAS|FAPIIS|FAC\n'+ 
                            'Data_Load_API.py [Environment] [Data file type].\n'+
                            'Example: Data_Load_API.py Test SAM ')
        sys.exit(1)
    v_fiscal_year = None
    if (v_data_name in ['USAS','FAC']):
        try:
            v_fiscal_year = sys.argv[3]
        except IndexError:
            v_fiscal_year = None
            print ('No fiscal year parameter has been provided.\n'+
                    'Latest fiscal year will be downloaded and imported.')

    main(v_db_env, v_data_name,v_fiscal_year)