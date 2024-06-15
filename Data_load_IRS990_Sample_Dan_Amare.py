from irsx.xmlrunner import XMLRunner
from datetime import datetime
from logging import getLogger
from irsx.settings import INDEX_DIRECTORY
from irsx.settings import WORKING_DIRECTORY
from irsx.settings import IRSX_CACHE_DIRECTORY
import csv
import os
import numpy as np
import pandas as pd
import re as re
import subprocess as sb
import sys 
import shutil
from datetime import date
import cx_Oracle
import requests
from requests.exceptions import HTTPError
import urllib3
from urllib3 import request
import time
import zipfile
import io
import glob
import zipfile_deflate64 as zipfile
from bs4 import BeautifulSoup



#This python script uses BeautifulSoup and IRSX libraries to pull and parse through IRS 990 filings available in xml format on IRS.gov.
#It pulls an entire year's worth of filings for grant award recipient organizations monitored by a federal agency.
# It also pulls additional 990 filings and audits from non-profit ProPublica website using APIs available to the public.
# The purpose is to obtain financial data used in risk analytics applications further downstream.
# THIS SCRIPT IS FOR DEMO ONLY, IT DOES NOT WORK AS ESSENTIAL PARAMETERS, CONFIG files, folders are made inaccessible.
# Entire script implemented by Dagnaw Amare, 2022-2024. 


logger = getLogger('return')


def combine_fields(value1, value2):
    '''Returns the sum of value1 and value2 or whichever operand is not None. Returns None when both are None.'''
    if value1 is not None and value2 is not None:
        return value1 + value2
    elif value1 is None:
        return value2
    elif value2 is None:
        return value1
    else:
        return None


'''The Return class represents shared attributes across all 990 forms. In general, these are values found in IRSx's ReturnHeader990x schedule.'''
class Return:
    def flatten_atts_for_db(self):
        '''
        Returns a flattened list of dictionaries populated with return header, balance, comp information for each person.
        '''
        # obj_tbl_field_map is a dictionary that maps object key names (ex: self.header_dict['ein']) to preferred field names on any output
        # flatten_atts_for_db() searches self.obj_tbl_field_map for user-defined custom keys before falling back to class' default keys
        db_rows = []
        for person in self.people:
            procd_person = {}
            for k, v in person.items():
                try:
                    k = self.obj_tbl_field_map[k]
                except (KeyError, TypeError) as e:
                    # logger.debug(e)
                    pass
                procd_person[k] = v
            for k, v in self.balance_dict.items():
                try:
                    k = self.obj_tbl_field_map[k]
                except (KeyError, TypeError) as e:
                    # logger.debug(e)
                    pass
                procd_person[k] = v
            for k, v in self.header_dict.items():
                try:
                    k = self.obj_tbl_field_map[k]
                except (KeyError, TypeError) as e:
                    # logger.debug(e)
                    pass
                procd_person[k] = v
            procd_person['object_id'] = self.object_id
            db_rows.append(procd_person)

        return db_rows

    def process_header_fields(self):
        '''
        Process header information from ReturnHeader990x and return it to __init__ as a dict
        Header information will be handled the same across all forms of the 990
        '''
        header = self.xml_runner.run_sked(self.object_id, 'ReturnHeader990x')
        header_dict = {}
        try:
            results = header.get_result()[0]
        except TypeError:
            #print('Unsupport version %s'%self.object_id)
            return header_dict 

        header_values = results['schedule_parts']['returnheader990x_part_i']        

        header_obj_irsx_map = {
            'ein': 'ein',
            'name': 'BsnssNm_BsnssNmLn1Txt',
            'state': 'USAddrss_SttAbbrvtnCd',
            'city': 'USAddrss_CtyNm',
            'tax_year': 'RtrnHdr_TxYr'
            
            # custom_key_name: IRSX_key_name
            # add valid ReturnHeader990x keys here to save those values during processing (see variables.csv)
        }

        obj_str_handling_map = {
            'state': lambda x: x.upper(),
            'city': lambda x: x.title(),
            'name': lambda x: x.title()
            # map custom keys to string handling lambdas for quick & dirty cleaning

        }

        for obj_key, irsx_key in header_obj_irsx_map.items():
            try:
                value = header_values[irsx_key]
                if obj_key in obj_str_handling_map:
                    value = obj_str_handling_map[obj_key](value)
                header_dict[obj_key] = value
            except KeyError:
                header_dict[obj_key] = None

        # special case: fiscal year date
        try:
            tax_prd_end_date = header_values['RtrnHdr_TxPrdEndDt']
            tax_prd_end_date =  datetime.strptime(tax_prd_end_date, '%Y-%m-%d')
            header_dict['fiscal_year'] = tax_prd_end_date.year
        except KeyError:
            header_dict['fiscal_year'] = None
        
        try:
            header_dict['dba'] = header_values['BsnssNm_BsnssNmLn1Txt']
        except KeyError:
            header_dict['dba'] = None
        
        return header_dict

    def process_compensation_fields(self):
        '''Compensation fields are specific to the flavor of 990 and this is implemented on child classes.'''
        raise NotImplementedError

    def process_summary_fields(self):
        '''Balance fields are specific to the flavor of 990 and this is implemented on child classes.'''
        raise NotImplementedError
    def process_balance_fields(self):
        '''Balance fields are specific to the flavor of 990 and this is implemented on child classes.'''
        raise NotImplementedError
    def process_expense_fields(self):
        '''Balance fields are specific to the flavor of 990 and this is implemented on child classes.'''
        raise NotImplementedError
    def process_governance_fields(self):
        '''Balance fields are specific to the flavor of 990 and this is implemented on child classes.'''
        raise NotImplementedError
    def process_statement_fields(self):
        '''Balance fields are specific to the flavor of 990 and this is implemented on child classes.'''
        raise NotImplementedError

    def __init__(self, object_id,obj_tbl_field_map=None):
        self.object_id = object_id
        self.xml_runner = XMLRunner()
        self.obj_tbl_field_map = obj_tbl_field_map

        self.header_dict = self.process_header_fields()
        #if not header_only:
        self.summary_dict = self.process_summary_fields()
        self.balance_dict = self.process_balance_fields()
        self.expense_dict = self.process_expense_fields()
        self.governance_dict = self.process_governance_fields()
        self.statement_dict = self.process_statement_fields()
        
        #self.people = self.process_compensation_fields()

        self.failures = {
            #'comp': True if self.people is None else False,
            'header': True if self.header_dict is None else False,
            'summary': True if self.summary_dict is None else False,
            'balance': True if self.balance_dict is None else False,
            'expense': True if self.expense_dict is None else False,
            'governance': True if self.governance_dict is None else False,
            'statement': True if self.statement_dict is None else False,
        
            }
        


    def __repr__(self):
        return '''Object_ID: {object_id}\nHeader: {header}\nSummary: {summary}\n 
                    Balance: {balance}\n Expense: {expense}\n Governance: {governance}\n
                    Statement: {statement}\n '''.format(
                #People: {people}\n
              
            object_id=self.object_id,
            header=self.header_dict,
            summary=self.summary_dict,
            balance=self.balance_dict,
            expense = self.expense_dict,
            governance = self.governance_dict,
            statement = self.statement_dict
            #,
            #people=self.people
        )



'''This child class represents information we're pulling from the 990EO'''
class Return_990(Return):
    def process_compensation_fields(self):
        db_irsx_key_map = {
                 'person': 'PrsnNm',
                 'title': 'TtlTxt',
                 'base_org': 'BsCmpnstnFlngOrgAmt',
                 'base_rel': 'CmpnstnBsdOnRltdOrgsAmt',
                 'bonus_org': 'BnsFlngOrgnztnAmnt',
                 'bonus_rel': 'BnsRltdOrgnztnsAmt',
                 'other_org': 'OthrCmpnstnFlngOrgAmt',
                 'other_rel': 'OthrCmpnstnRltdOrgsAmt',
                 'defer_org': 'DfrrdCmpnstnFlngOrgAmt',
                 'defer_rel': 'DfrrdCmpRltdOrgsAmt',
                 'nontax_ben_org': 'NntxblBnftsFlngOrgAmt',
                 'nontax_ben_rel': 'NntxblBnftsRltdOrgsAmt',
                 '990_total_org': 'TtlCmpnstnFlngOrgAmt',
                 '990_total_rel': 'TtlCmpnstnRltdOrgsAmt',
                 'prev_rep_org': 'CmpRprtPrr990FlngOrgAmt',
                 'prev_rep_rel': 'CmpRprtPrr990RltdOrgsAmt'
                # custom_key_name: IRSX_key_name
                # add valid Schedule J IRSX key names here to save those values during processing (see variables.csv)
             }

        db_type_map = {
            # Key your custom keys to types to specify how each key should be cast on processing
            key: int for key in db_irsx_key_map
        }

        obj_str_handling_map = {
            # Key custom keys to lambda functions for quick and dirty cleaning on those values
            'person': lambda x: x.title(),
            'title': lambda x: x.title()
        }

        db_type_map['person'] = str
        db_type_map['title'] = str

        sked_j = self.xml_runner.run_sked(self.object_id, 'IRS990ScheduleJ')
        results = sked_j.get_result()[0]
        try:
            sked_j_values = results['groups']['SkdJRltdOrgOffcrTrstKyEmpl']
        except KeyError:
            return None

        people = []

        for employee_dict in sked_j_values:
            processed = {}
            for db_key in db_irsx_key_map:
                irsx_key = db_irsx_key_map[db_key]
                db_type = db_type_map[db_key]
                try:
                    value = db_type(employee_dict[irsx_key]) if irsx_key in employee_dict else None
                    if value is None and irsx_key == 'PrsnNm':
                        # sometimes people's names show up under BsnssNmLn1Txt
                        alt_person_key = 'BsnssNmLn1Txt'
                        value = employee_dict[alt_person_key] if alt_person_key in employee_dict.keys() else None
                    if db_key in obj_str_handling_map:
                        value = obj_str_handling_map[db_key](value)
                    processed[db_key] = value
                except TypeError:
                    # if we can't cast the value we set it to None
                    processed[db_key] = None
                except AttributeError:
                    processed[db_key] = None
            person_name = processed['person']

            people.append(processed)

        return people

    def process_summary_fields(self):

        #balance = self.xml_runner.run_sked(self.object_id, 'IRS990')
        parsed_filing =  self.xml_runner.run_filing(self.object_id)
        processed = {}
        
        result = parsed_filing.get_result()
        try:
            sked = result[0]
        except TypeError:
            return processed

        #schedule_list = parsed_filing.list_schedules()
        #parsed_skedez = parsed_filing.get_parsed_sked('IRS990EZ')[0] 
        
        for sked in result:
            if sked['schedule_name'] not in [ 'IRS990', 'IRS990EZ']:
                continue
            if sked['schedule_name'] == 'IRS990':
                results = parsed_filing.get_parsed_sked('IRS990')[0]
                db_irsx_key_map = {'total_rev': 'CYTtlRvnAmt' ,
                                   'total_exp': 'CYTtlExpnssAmt',
                                   'net_assets': 'NtAsstsOrFndBlncsEOYAmt',
                                   'grants_contrbtns':'CYCntrbtnsGrntsAmt',
                                    'service_rev':'CYPrgrmSrvcRvnAmt',
                                    'invstmnt_income':'CYInvstmntIncmAmt',
                                    'other_rev':'CYOthrRvnAmt',
                                    'Salaries_exp':'CYSlrsCmpEmpBnftPdAmt',
                                    'grants_paid_exp':'CYGrntsAndSmlrPdAmt',
                                    'tot_fundraising_exp':'CYTtlFndrsngExpnsAmt',
                                    'other_exp':'CYOthrExpnssAmt',
                                    'Num_staff':'TtlEmplyCnt'
                                   }
                part_name = 'part_i'
                #print("Schedule: %s" % sked['schedule_name'])
                
            if sked['schedule_name'] == 'IRS990EZ':
                results = parsed_filing.get_parsed_sked('IRS990EZ')[0]
                db_irsx_key_map = { 'total_rev': 'TtlRvnAmt',
                                   'total_exp': 'TtlExpnssAmt',
                                   'net_assets':'NtAsstsOrFndBlncsEOYAmt',
                                   'grants_contrbtns':'CntrbtnsGftsGrntsEtcAmt',
                                    'service_rev':'PrgrmSrvcRvnAmt',
                                    'invstmnt_income':'InvstmntIncmAmt',
                                    'other_rev':'OthrRvnTtlAmt',
                                    'Salaries_exp':'SlrsOthrCmpEmplBnftAmt',
                                    'grants_paid_exp':'GrntsAndSmlrAmntsPdAmt',
                                    'tot_fundraising_exp':'SpclEvntsDrctExpnssAmt',
                                    'other_exp':'OthrExpnssTtlAmt'
                                    #'Num_staff':''
                                   
                                   }
                part_name = 'ez_part_i'
                #print("Schedule: %s" % sked['schedule_name'])
            
            
        #print("Revenue: %s" % parsed_skedez['schedule_parts']['ez_part_i']['TtlRvnAmt'])
        #results = balance.get_result()[0]

            try:
                #balance_values = results['schedule_parts']['ez_part_i']
                values = results['schedule_parts'][part_name]
            except KeyError:
                return None

            
            for db_key in db_irsx_key_map:
                irsx_key = db_irsx_key_map[db_key]
                processed[db_key] = values[irsx_key] if irsx_key in values.keys() else None

                try:
                    processed[db_key] = int(processed[db_key])
                except TypeError:
                    processed[db_key] = None

        #p = processed
        #p['private_support'] = combine_fields(p['total_contrib'], p['govt_grants'])

        return processed
    def process_balance_fields(self):

        #balance = self.xml_runner.run_sked(self.object_id, 'IRS990')
        parsed_filing =  self.xml_runner.run_filing(self.object_id)
        #result = parsed_filing.get_result()
        #schedule_list = parsed_filing.list_schedules()
        #parsed_skedez = parsed_filing.get_parsed_sked('IRS990EZ')[0] 
        processed = {}
        try:
            tax_year = self.header_dict['tax_year']
        except KeyError:
            print('Tax year key error for obect id %s'%self.object_id)    

        try:
            ein = self.header_dict['ein']
        except KeyError:
            print('EIN key error for obect id %s'%self.object_id)    

        object_id = self.object_id
        
        #print(tax_year)
        #Unrestrictedassets = 'UnrstrctdNtAssts_EOYAmt' if year<2019 else 'UnrstrctdNtAssts_EOYAmt' 
        result = parsed_filing.get_result()
        try:
            sked = result[0]
        except TypeError:
            return processed

        for sked in result:
            if sked['schedule_name'] not in [ 'IRS990','IRS990EZ']:
                continue

            if sked['schedule_name'] == 'IRS990':
                results = parsed_filing.get_parsed_sked('IRS990')[0]
                db_irsx_key_map = {
                                'Un_net_assets':'UnrstrctdNtAssts_EOYAmt',
                                'total_assets':'TtlAssts_EOYAmt',
                                'total_liabilities':'TtlLblts_EOYAmt',                                
                                'Cash_BOY':'CshNnIntrstBrng_BOYAmt',
                                'Cash_EOY':'CshNnIntrstBrng_EOYAmt',
                                'Savings_BOY':'SvngsAndTmpCshInvst_BOYAmt',
                                'Savings_EOY':'SvngsAndTmpCshInvst_EOYAmt',
                                'grants_receivable':'PldgsAndGrntsRcvbl_EOYAmt',
                                'accts_receivable':'AccntsRcvbl_EOYAmt',
                                'investments':'InvstmntsPrgrmRltd_EOYAmt',
                                'land_bldg_equip':'LndBldgEqpCstOrOthrBssAmt'
                                
                                   }
                part_name = 'part_x'
            
            if sked['schedule_name'] == 'IRS990EZ':
                results = parsed_filing.get_parsed_sked('IRS990EZ')[0]
                db_irsx_key_map = {#'Un_net_assets':'Na',
                                'total_assets':'Frm990TtlAssts_EOYAmt',
                                'total_liabilities':'SmOfTtlLblts_EOYAmt',                                
                                'Cash_BOY':'CshSvngsAndInvstmnts_BOYAmt',
                                'Cash_EOY':'CshSvngsAndInvstmnts_EOYAmt',
                                #'Savings_BOY':'Na',
                                #'Savings_EOY':'Na'
                                'Savings_BOY':None,
                                'Savings_EOY':None,
                                'grants_receivable':None,
                                'accts_receivable':None,
                                'investments':'CshSvngsAndInvstmnts_EOYAmt',
                                'land_bldg_equip':'LndAndBldngs_EOYAmt'                        
                                   }
                part_name = 'ez_part_ii'            
                
            
            
            try:
                values = results['schedule_parts'][part_name]            
                #if (ein=='630590338'):
                    #print('For Ein %s Part keys are: %s'%(ein,values.keys()))
                    #print('Ein %s has object id %s for tax year %s'%(ein, object_id, tax_year))
                    #exit(0)
                
            except KeyError:
                return None
            
            for db_key in db_irsx_key_map:
                irsx_key = db_irsx_key_map[db_key]
                processed[db_key] = values[irsx_key] if irsx_key in values.keys() else None
                try:
                    processed[db_key] = int(processed[db_key])
                except TypeError:
                    processed[db_key] = None

        return processed
    def process_expense_fields(self):

        #balance = self.xml_runner.run_sked(self.object_id, 'IRS990')
        parsed_filing =  self.xml_runner.run_filing(self.object_id)
        #result = parsed_filing.get_result()
        #schedule_list = parsed_filing.list_schedules()
        #parsed_skedez = parsed_filing.get_parsed_sked('IRS990EZ')[0] 
        processed = {}
        result = parsed_filing.get_result()
        try:
            sked = result[0]
        except TypeError:
            return processed

        db_irsx_key_map = {'Depreciation': 'DprctnDpltn_TtlAmt' }
                               
        for sked in result:
            if sked['schedule_name'] not in [ 'IRS990']:
                continue

            results = parsed_filing.get_parsed_sked('IRS990')[0]
                
            try:
                #balance_values = results['schedule_parts']['ez_part_i']
                values = results['schedule_parts']['part_ix']            
            except KeyError:
                return None
            
            
            for db_key in db_irsx_key_map:
                irsx_key = db_irsx_key_map[db_key]
                processed[db_key] = values[irsx_key] if irsx_key in values.keys() else None

                try:
                    processed[db_key] = int(processed[db_key])
                except TypeError:
                    processed[db_key] = None

        return processed
    def process_governance_fields(self):

        #balance = self.xml_runner.run_sked(self.object_id, 'IRS990')
        parsed_filing =  self.xml_runner.run_filing(self.object_id)
        #result = parsed_filing.get_result()
        #schedule_list = parsed_filing.list_schedules()
        #parsed_skedez = parsed_filing.get_parsed_sked('IRS990EZ')[0] 
        processed = {}
        
        result = parsed_filing.get_result()
        try:
            sked = result[0]
        except TypeError:
            return processed        
        
        db_irsx_key_map = {'Conflict_int': 'CnflctOfIntrstPlcyInd',
                                'Disclose_int': 'AnnlDsclsrCvrdPrsnInd',
                                'Monitor_complnce':'RglrMntrngEnfrcInd',
                                'Diversion_assets':'MtrlDvrsnOrMssInd'}
        string_map = {'true': 'Yes',
                        'false': 'No',
                        '1': 'Yes',
                        '0': 'No'}                                
                               
        for sked in result:
            if sked['schedule_name'] not in [ 'IRS990']:
                continue

            results = parsed_filing.get_parsed_sked('IRS990')[0]
                
            try:
                values = results['schedule_parts']['part_vi']            
            except KeyError:
                return None
            
            
            for db_key in db_irsx_key_map:
                irsx_key = db_irsx_key_map[db_key]
                processed[db_key] = values[irsx_key] if irsx_key in values.keys() else None

                try:
                    processed[db_key] = string_map[(processed[db_key])]
                except KeyError:
                    processed[db_key] = None

        return processed
    def process_statement_fields(self):

        #balance = self.xml_runner.run_sked(self.object_id, 'IRS990')
        parsed_filing =  self.xml_runner.run_filing(self.object_id)
        #result = parsed_filing.get_result()
        #schedule_list = parsed_filing.list_schedules()
        #parsed_skedez = parsed_filing.get_parsed_sked('IRS990EZ')[0] 
        processed = {}
        
        result = parsed_filing.get_result()
        try:
            sked = result[0]
        except TypeError:
            return processed

        db_irsx_key_map = {'Audit_req': 'FdrlGrntAdtRqrdInd',
                                'Audit_done': 'FdrlGrntAdtPrfrmdInd'
                                }
        string_map = {'true': 'Yes',
                        'false': 'No',
                        '1': 'Yes',
                        '0': 'No'}
                               
        for sked in result:
            if sked['schedule_name'] not in [ 'IRS990']:
                continue

            results = parsed_filing.get_parsed_sked('IRS990')[0]
                
            try:
                values = results['schedule_parts']['part_xii']            
            except KeyError:
                return None
            
            
            for db_key in db_irsx_key_map:
                irsx_key = db_irsx_key_map[db_key]
                processed[db_key] = values[irsx_key] if irsx_key in values.keys() else None

                try:
                    processed[db_key] = string_map[(processed[db_key])]
                except KeyError:
                    processed[db_key] = None

        return processed

def Get_cngrants_data(v_db_env,LOG_FILE):
    lib_dir = r"C:\oracle\Product\instantclient_21_3"
    
    try:
        cx_Oracle.init_oracle_client(lib_dir=lib_dir)
    except Exception as err:
        print_txt = "Error connecting: cx_Oracle.init_oracle_client()\n"
        print_txt = print_txt + err
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
    
    db_dsn = 'Connection information'
    try:
        connection = cx_Oracle.connect(user=user_id, password=passwd,
                               dsn=db_dsn)
    except Exception as err:
        print_txt='Error connecting: Invalid User/Passowrd. Please check and try again.'
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)

    cursor = connection.cursor()    
    sql = """  select distinct trim(mu.org_id) as org_id, ein 
                from ares.mv_daily_app_universe mu, cspanowner.organizations o
                where mu.org_id = o.id
                and o.ein is not null 
                """

    cursor.execute(sql)
    col_names = ['ORG_ID','EIN']
    cngrants_data = pd.DataFrame(cursor.fetchall(),columns=col_names)
    print_txt = 'Org_id and EIN in Monitoring/Application Universe: %s'%cursor.rowcount
    print_log(LOG_FILE, print_txt,1)
    
    #Drop non integer EINs
    numeric = lambda x:int(x) if x.isnumeric() else 0
    cngrants_data['EIN'] = [numeric(x) for x in cngrants_data['EIN'] ]
    bad_eins = len(cngrants_data[cngrants_data['EIN'] == 0])
    print_txt = 'Delete Non numberic EINs in Monitoring/Application Universe: %s'% bad_eins
    print_log(LOG_FILE, print_txt,1)
    
    cngrants_data = cngrants_data[cngrants_data['EIN'] != 0]
    connection.close()

    return cngrants_data

def Import_filings(v_db_env,table_name,df_filings,filing_headers,LOG_FILE):
    #lib_dir = r"C:\oracle\Product\instantclient_21_3"
    
    #try:
        #cx_Oracle.init_oracle_client(lib_dir=lib_dir)
    #except Exception as err:
        #print("Error connecting: cx_Oracle.init_oracle_client()")
        #print(err);
        #sys.exit(1);

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
    db_dsn = 'Conn_dsn'
    try:
        connection = cx_Oracle.connect(user=user_id, password=passwd,
                               dsn=db_dsn)
    except Exception as err:
        print_txt='Error connecting: Invalid User/Passowrd. Please check and try again.'
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)

    cursor = connection.cursor()    
    col_name = filing_headers
    col_name_str = """ ("""+','.join(col_name)+""") """
    #vals = ['23456','Test Organization Name']
    #f = lambda x: "N/A" if pd.isna(x) else ("" if pd.isnull(x) else str(x))
    f = lambda x: "" if pd.isnull(x) else str(x)
    vals_vars = """ (:"""+',:'.join(col_name)+""")"""
    #sql = """ insert into """+ra_table+""" (application_id) values( :did)"""
    #sql = """ insert into ARES.RA_IRS990"""+col_name_str + """values""" + vals_vars      
    sql = """ insert into ARES."""+table_name+col_name_str + """values""" + vals_vars  
    rowcount = 0
    #print('Insert sql is: %s'%sql)
    
    file_recs = len(df_filings.index)
    increment = 100 if int(file_recs/5) < 100 else int(file_recs/5) 
    increment = int(increment/100)*100
    for index, rows in df_filings.iterrows():       
        vals = [f(x) for x in rows]        
        #print('Values are: %s'%vals)
        try:
            cursor.execute(sql,vals)
        except Exception as e:
            error, = e.args
            if error.code in (12899,1438):
                print ('Value too large for Column, skipping...')
                continue
            else:
                print (error.code)
                print(error.message)
                print(error.context)
                exit(1)
        rowcount += 1
        if (rowcount%increment == 0):
            print('Inserted records %s/%s ...'%(rowcount,file_recs))

    #sql = """ UPDATE ARES.RA_IRS990 SET CREATE_DT = SYSDATE WHERE CREATE_DT IS NULL"""  

    sql = """ UPDATE ARES."""+table_name+""" SET CREATE_DT = SYSDATE
                WHERE CREATE_DT IS NULL"""                  
    #print('Update create dt sql is: %s'%sql)
    cursor.execute(sql)

    connection.commit()
    print_txt='Successfully imported records into %s: %s/%s'%(table_name,rowcount,file_recs)
    print_log(LOG_FILE, print_txt,1)
    

    connection.close()

def Download_index_file(v_index_year, LOG_FILE):
    
    
    today = datetime.now().strftime("%m_%d_%Y_%H_%M")
    
    index_year = str(v_index_year)
    index_csv = 'index_'+index_year+'.csv'
    index_csv_bkp = 'index_'+index_year+'_'+today+'.csv'
    INDEX_FILE= os.path.join(INDEX_DIRECTORY, index_csv)
    INDEX_FILE_BKP= os.path.join(INDEX_DIRECTORY, index_csv_bkp)
    
    if os.path.exists(INDEX_FILE):
        print_txt='Copying Index file %s to archive before downloading...' %(index_year)
        print_log(LOG_FILE, print_txt,1)
        shutil.copy( INDEX_FILE,INDEX_FILE_BKP )
    
    
    print_txt='Downloading Index file for filing year %s...' %(index_year)
    print_log(LOG_FILE, print_txt,1)
    
    #Download steps here, previously AWS code 
    #https://apps.irs.gov/pub/epostcard/990/xml/2019/index_2019.csv
    #sb.run(['irsx_index', '--year='+index_year])
    REMOTE_DOWNLOAD_SITE = r'https://apps.irs.gov/pub/epostcard/990/xml/'+index_year+r'/'
    INDEX_FILE_URL = REMOTE_DOWNLOAD_SITE+index_csv
    print ('Index url: %s'%INDEX_FILE_URL)
    res = requests.head(INDEX_FILE_URL)
    if res.status_code != 200:
        print_txt='Index file is not available for download %s...' %(INDEX_FILE_URL)
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)
    #Download to user default directory
    users_dir = r'C:\Users'
    username = os.getlogin()
    LOCAL_DOWNLOAD_DIRECTORY = os.path.join(users_dir,username,'Downloads')
    r = requests.get(INDEX_FILE_URL)    
    if r.status_code != 200:
        print_txt='Failed to download index file: %s '%(INDEX_FILE_URL)
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)    

    INDEX_FILE_NEW = os.path.join(LOCAL_DOWNLOAD_DIRECTORY,index_csv)
    open(INDEX_FILE_NEW,'wb').write(r.content)        
    print_txt='Successfully downloaded index file to: %s '%(INDEX_FILE_NEW)
    print_log(LOG_FILE, print_txt,1)
    
    #Copy from Download to Index CSV directory
    if os.path.exists(INDEX_FILE_NEW):
        print_txt='Copying New Index file %s to CSV directory...' %(index_year)
        print_log(LOG_FILE, print_txt,1)
        shutil.copy( INDEX_FILE_NEW,INDEX_FILE )


    """
    Previous code to generate Index file from XML files...

    #For each file in XML directory, get OBJECT ID and corresponding EIN
    index_list = []
    num_files = len(os.listdir(WORKING_DIRECTORY))
    proc_files = 0
    for xmlfile in os.listdir(WORKING_DIRECTORY):
        object_id = xmlfile.split('_')[0]
        #ein = XMLRunner().run_sked(object_id, 'ReturnHeader990x').get_ein()
        
        header_dict = Return_990(object_id,True).header_dict
        try:
            ein = header_dict['ein']
            name = header_dict['name']
            #tax_year = header_dict['tax_year']
            #fiscal_year = header_dict['fiscal_year']
                        
        except KeyError:
            ein = None
        #index_list.append([ein, object_id,name, tax_year,fiscal_year])        
        index_list.append([ein, object_id,name])
        #index_list.append([ein, object_id,'test org'])
        proc_files += 1
        if (proc_files%5000==0):
            print('Number of xml files processed to generate Index file: %s/%s'%(proc_files,num_files))
            break
        #if num_files > 10:
            #break
    #print('Number of xml efiles use to generate Index file: %s'%num_files)

    df_index = pd.DataFrame(index_list,columns =['EIN','OBJECT_ID','Name'])
    #filings_csv = 'Test_filings_'+str(index_year)+'.csv'                
    #FILINGS_OUTPUT = os.path.join( CURRENT_DIRECTORY, filings_csv)
    df_index.to_csv(INDEX_FILE, index=False)
    """
        
    print_txt='Index file downloaded for %s filing year.\nIndex file location:\n %s'%(index_year, INDEX_FILE)
    print_log(LOG_FILE, print_txt,1)
    
    return INDEX_FILE


def Prepare_xml_files(LOCAL_ZIP_FILE, local_file,df_efilers, LOG_FILE):

    ARCHIVE_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'Archive') 
    #Process zip file
    #z = zipfile.ZipFile(io.BytesIO(LOCAL_ZIP_FILE))
    z = zipfile.ZipFile(LOCAL_ZIP_FILE,'r')

    print_txt = "Processing IRS XML zip file: %s" %(local_file) 
    print_log(LOG_FILE, print_txt,1)
    
    #DATA_FILE_NAME = z.namelist()[0]
    ARCHIVE_ZIP = os.path.join(ARCHIVE_DIRECTORY, local_file) 
    #Clean archive directory
    for xmlfile in glob.iglob(os.path.join(ARCHIVE_DIRECTORY, '*.xml')):
        os.remove(xmlfile)
    #Extract into archive directory       
    print_txt = "Extracting IRS XML file: %s" %(LOCAL_ZIP_FILE) 
    print_log(LOG_FILE, print_txt,1)
    

    
    try:
        z.extractall(ARCHIVE_DIRECTORY)
    except RuntimeError as e:
        print_txt = "Error extracting zip file file: %s\n" %(LOCAL_ZIP_FILE) 
        print_txt = print_txt + e
        print_log(LOG_FILE, print_txt,1)
        exit(1)


    print_txt = "Extraction IRS XML file completed for: %s" %(LOCAL_ZIP_FILE) 
    print_log(LOG_FILE, print_txt,1)
    
    #Copy XML files needed to Working directory
    print_txt = "Copying needed IRS XML files to: %s" %(WORKING_DIRECTORY) 
    print_log(LOG_FILE, print_txt,1)
        
    #Get object id for xmlfile in os.listdir(WORKING_DIRECTORY):
    list_objectids = []
    for xmlfile in glob.iglob(os.path.join(ARCHIVE_DIRECTORY, '*.xml')):
        filename = os.path.basename(xmlfile)        
        #print ('firstfile is %s'%filename)            
        object_id = filename.split('_')[0]
        list_objectids.append(object_id)
    df_objectids = pd.DataFrame(list_objectids, columns=['OBJECT_ID'])
    #Collect object ids that have corresponding XML files
    df_objectids_found = pd.merge(df_efilers,
                 df_objectids,
                 on='OBJECT_ID', how='inner')['OBJECT_ID']
    num_objectids_found = len(df_objectids_found.index)  
    num_xmls = len(df_objectids.index)           
    for object_id in list(df_objectids_found):
        xmlfile = object_id+'_public.xml'
        xmlfilepath = os.path.join(ARCHIVE_DIRECTORY,xmlfile ) 
        shutil.copy(xmlfilepath,WORKING_DIRECTORY)

    
    #remove XML files not needed
    print_txt = "Found %s/%s needed IRS XML files from ZIP file %s, removing the rest..." %(num_objectids_found, num_xmls, local_file) 
    print_log(LOG_FILE, print_txt,1)
    for xmlfile in glob.iglob(os.path.join(ARCHIVE_DIRECTORY, '*.xml')):
        os.remove(xmlfile)
                                
    #If not archived already, copy to archive 
    if not os.path.exists(ARCHIVE_ZIP):
        print_txt='Archiving zipped file to %s' %(ARCHIVE_ZIP)
        print_log(LOG_FILE, print_txt,1)
        shutil.copy( LOCAL_ZIP_FILE,ARCHIVE_ZIP )
        #shutil.copy( r.content,ARCHIVE_ZIP )
        #shutil.copyfileobj(mp3file, output)
    return

def listFD(url, ext=''):
    page = requests.get(url).content
    #print(page[:1000])
    soup = BeautifulSoup(page, 'html.parser')
    
    #return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]  
    #return [node.get('title') for node in soup.find_all('a', attrs = {'class':'link-label label-file label-file-zip'}) ]
    return [node.get('href') for node in soup.find_all('a') if node.get('href') is not None and node.get('href').startswith(ext) ]


def Get_names_files(index_year, LOG_FILE):    
        
    REMOTE_DOWNLOAD_SITE = r'https://apps.irs.gov/pub/epostcard/990/xml/'+index_year+r'/'
    #ZIP_FILE_URL = REMOTE_DOWNLOAD_SITE #+remote_file
        

    url = 'https://www.irs.gov/charities-non-profits/form-990-series-downloads'
    ext = REMOTE_DOWNLOAD_SITE

    all_files = listFD(url, ext)
    zip_files = [file for file in all_files if file.endswith('zip')]
    file_sizes = []
    for file in zip_files:
        res = requests.head(file)
        if res.status_code == 200:
            #For future download progress use, use download in chunks code
            file_sizes.append(res.headers['content-length'])
            print_txt='File available to download: %s'%file
            print_log(LOG_FILE, print_txt,1)            
            
    print_txt='Total number of zip files to download: %s'%(len(zip_files))
    print_log(LOG_FILE, print_txt,1)    
      

    return zip_files

def Download_zip_files(index_year, file_names, LOG_FILE):
    """
    docstring
    """
    #pass
    #User default download
    users_dir = r'C:\Users'
    username = os.getlogin()
    LOCAL_DOWNLOAD_DIRECTORY = os.path.join(users_dir,username,'Downloads')
    #Download files only if not previously downloaded.
    for file in file_names:

        remote_file = os.path.basename(file)
        #print('Remote file name is %s'%remote_file)
        LOCAL_ZIP_FILE = os.path.join(LOCAL_DOWNLOAD_DIRECTORY,remote_file)
        #Skip file already downloaded        
        if os.path.exists(LOCAL_ZIP_FILE):
            print_txt='IRS ZIP file already downloaded: %s '%(LOCAL_ZIP_FILE)
            print_log(LOG_FILE, print_txt,1)
            continue
        
        print_txt='Downloading IRS ZIP file to: %s '%(LOCAL_ZIP_FILE)
        print_log(LOG_FILE, print_txt,1)
        
        r = requests.get(file)    
        if r.status_code != 200:
            print_txt='Download IRS ZIP file failed for: %s\n'%(LOCAL_ZIP_FILE)
            print_txt=print_txt + 'Please try again later.'            
            print_log(LOG_FILE, print_txt,1)
            sys.exit(1)    

        """
        response = requests.get(url, stream=True)
        with open('alaska.zip', "wb") as f:
        for chunk in response.iter_content(chunk_size=512):
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
        """            
        open(LOCAL_ZIP_FILE,'wb').write(r.content)        
        print_txt='Successfully downloaded IRS ZIP file to: %s '%(LOCAL_ZIP_FILE)
        print_log(LOG_FILE, print_txt,1)
        
    return
def get_irsgov_data(v_index_year, df_cngrants, LOG_FILE):

    #Download index file and load into Dataframe
    INDEX_FILE = Download_index_file(v_index_year, LOG_FILE)
    col_names = ['RETURN_ID','FILING_TYPE','EIN','TAX_PERIOD','SUB_DATE','TAXPAYER_NAME','RETURN_TYPE','DLN','OBJECT_ID']
    #col_names = ['EIN','OBJECT_ID','Name']
    np_year = pd.read_csv(INDEX_FILE, names=col_names, converters = {'OBJECT_ID': str,'EIN':str}, header=0,encoding='ISO-8859-1', index_col=False, low_memory=False)

    CURRENT_DIRECTORY = os.getcwd()
    #Get CN_Grants EINs only from Index data
    df_cngrants_efilers = pd.merge(np_year,df_cngrants,on='EIN', how='inner')              
    
    #extract latest if duplicate/ammendement filing in same year
    #df_cngrants_efilers = df_cngrants_efilers.sort_values(by=['ORG_ID','TAX_PERIOD','RETURN_ID'])
    #df_cngrants_efilers = df_cngrants_efilers.drop_duplicates(['ORG_ID','TAX_PERIOD'], keep='last')
    #test convert to string OBJECT_ID scientific notation
    #df_cngrants_efilers['OBJECT_ID'] = [np.format_float_positidonal(x)[:-1] for x in df_cngrants_efilers['OBJECT_ID']]    
    #df_cngrants_efilers['OBJECT_ID'] = [str(x) for x in df_cngrants_efilers['OBJECT_ID']]
    index_year = str(v_index_year)
    cngrants_orgs = len(df_cngrants)    
    efiles_found = len(df_cngrants_efilers)
    print_txt='Found a total of %d/%d CNCS Orgs filings in Index file for %s efilers'%(efiles_found,cngrants_orgs, index_year )
    print_log(LOG_FILE, print_txt,1)
    
    print_txt='Processing %d/%d XML zip files for %s efilers'%(efiles_found,cngrants_orgs, index_year )
    print_log(LOG_FILE, print_txt,1)
    #User default download
    users_dir = r'C:\Users'
    username = os.getlogin()
    LOCAL_DOWNLOAD_DIRECTORY = os.path.join(users_dir,username,'Downloads')
    #Determine number of remote files available for download
    file_names = Get_names_files(index_year, LOG_FILE)
    Download_zip_files(index_year, file_names, LOG_FILE)                
    #Clean up WORKDING_DIRECTORY first, remove XML files from previous job runs
    for xmlfile in glob.iglob(os.path.join(WORKING_DIRECTORY, '*.xml')):
        os.remove(xmlfile)

    #For each downloaded zip file, extract to archive, check against CNCS index and save for later process
    df_efilers = df_cngrants_efilers.copy()
    
    for file in file_names:

        local_file = os.path.basename(file) 
        LOCAL_ZIP_FILE = os.path.join(LOCAL_DOWNLOAD_DIRECTORY,local_file)
        ARCHIVE_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'Archive')
        ARCHIVE_ZIP = os.path.join(ARCHIVE_DIRECTORY, local_file) 
        #Process if archive file doesn't exist, process anyway during testing
        #if not os.path.exists(ARCHIVE_ZIP):        
        #if os.path.exists(ARCHIVE_ZIP):
        #Extract file to Archive, copy into working directory files needed
        
        Prepare_xml_files(LOCAL_ZIP_FILE,local_file,df_efilers,LOG_FILE )
    

    #Process only XML files found in downloaded zip files
    list_xmls = []
    for xmlfile in glob.iglob(os.path.join(WORKING_DIRECTORY, '*.xml')):
        filename = os.path.basename(xmlfile)
        #print ('firstfile is %s'%filename)            
        object_id = filename.split('_')[0]                  
        list_xmls.append(object_id)
    df_xmls_found = pd.DataFrame(list_xmls,columns=['OBJECT_ID'])    
    #df_xmls_found = df_xmls_found.drop_duplicates(['OBJECT_ID'], keep='last')
    #Determine list of object_ids with missing XML files
    df_xmls_found['LJOIN'] = '1'
    df_objectids_missing = pd.merge(df_cngrants_efilers,
                 df_xmls_found,
                 on='OBJECT_ID', how='left') 
    df_objectids_missing = df_objectids_missing[df_objectids_missing['LJOIN'].isnull()]
    #df_objectids_missing = df_objectids_missing.drop(['LJOIN'],axis=1)
    
    #missing_objectids_csv = 'Missing_filings_'+str(index_year)+'.csv'                
    #MISSING_OUTPUT = os.path.join( CURRENT_DIRECTORY, missing_objectids_csv)
    #df_objectids_missing.to_csv(MISSING_OUTPUT, index=False)
    #Look for missing xmls in previous year archive files    
    efiles_missing_curr = len(df_objectids_missing)
    if efiles_missing_curr > 0:
        cngrants_orgs = len(df_cngrants_efilers)
        df_efilers =  df_objectids_missing.copy()
        index_year_prev = str(int(index_year)-1)
        print_txt='Unable to find %d/%d XML files, searching in previous year %s zip files.'%(efiles_missing_curr,cngrants_orgs, index_year_prev )
        print_log(LOG_FILE, print_txt,1)
        #Determine number of remote files available for download
        file_names = Get_names_files(index_year_prev, LOG_FILE)
        Download_zip_files(index_year_prev, file_names, LOG_FILE)                

        for file in file_names:

            local_file = os.path.basename(file) 
            LOCAL_ZIP_FILE = os.path.join(LOCAL_DOWNLOAD_DIRECTORY,local_file)
            ARCHIVE_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'Archive')
            ARCHIVE_ZIP = os.path.join(ARCHIVE_DIRECTORY, local_file) 

            Prepare_xml_files(LOCAL_ZIP_FILE,local_file,df_efilers,LOG_FILE )
    
        #Find any remaining XMLs not found
        list_xmls = []
        for xmlfile in glob.iglob(os.path.join(WORKING_DIRECTORY, '*.xml')):
            filename = os.path.basename(xmlfile)
            #print ('firstfile is %s'%filename)            
            object_id = filename.split('_')[0]                  
            list_xmls.append(object_id)
        df_xmls_found = pd.DataFrame(list_xmls,columns=['OBJECT_ID'])    
        #df_xmls_found = df_xmls_found.drop_duplicates(['OBJECT_ID'], keep='last')
        #Determine list of object_ids with missing XML files
        df_xmls_found['LJOIN'] = '1'
        df_objectids_missing = pd.merge(df_cngrants_efilers,
                 df_xmls_found,
                 on='OBJECT_ID', how='left') 
        df_objectids_missing = df_objectids_missing[df_objectids_missing['LJOIN'].isnull()]
        
    
        if len(df_objectids_missing) > 0:
            print_txt='Unable to find %d/%d missing XML files in previous year %s efilers'%(len(df_objectids_missing),efiles_missing_curr, index_year_prev )
            print_log(LOG_FILE, print_txt,1)
            
        else:
            print_txt='Found all of %d missing XML files in previous year %s efilers'%(efiles_missing_curr, index_year_prev )
            print_log(LOG_FILE, print_txt,1)
            
        df_cngrants_efilers = pd.merge(df_cngrants_efilers,
                 df_xmls_found,
                 on='OBJECT_ID', how='inner')
        df_cngrants_efilers = df_cngrants_efilers.drop(['LJOIN'],axis=1)
    
    efiles_found = len(df_cngrants_efilers)
    cngrants_orgs = len(df_cngrants)
    print_txt='Parsing through %d/%d XML files for %s efilers'%(efiles_found,cngrants_orgs, index_year )
    print_log(LOG_FILE, print_txt,1)
    
    #exit(0)
    #Test report for actual XML files to be parsed and loaded
    #filings_csv = 'Test_filings_'+str(index_year)+'.csv'                
    #FILINGS_OUTPUT = os.path.join( CURRENT_DIRECTORY, filings_csv)
    #df_cngrants_efilers.to_csv(FILINGS_OUTPUT, index=False)
    filings = []
    num_rows = 0
    DEMO_MAX = 20000
    for index, rows in df_cngrants_efilers.iterrows():
        num_rows += 1
        object_id = rows.OBJECT_ID
        org_id = rows.ORG_ID
        #object_id = '202043129349300954'
        r = Return_990(object_id)
        object_id_dict = {'OBJECT_ID':r.object_id,
                          'ORG_ID':org_id}
        #print(r.object_id)
        try:
            filings.append({**object_id_dict, **r.header_dict, **r.summary_dict, **r.balance_dict, 
                            **r.expense_dict, **r.governance_dict,**r.statement_dict})
        except TypeError as e:
            print_txt='Error parsing through filing for Object Id %s Org Id %s Fiscal Year %s.\n'%(object_id,org_id, index_year )
            print_txt=print_txt + 'Exception message is: %s'%str(e)
            print_log(LOG_FILE, print_txt,1)
            continue

        #print(r.flatten_atts_for_db())
        if(num_rows > DEMO_MAX):
            break
        if num_rows%100==0:
            #exit(0)
            print("Extracted filings %s/%s ..." %(num_rows,efiles_found))
    
    df_filings = pd.DataFrame(filings)
    #filings_csv = 'IRSGOV_filings_'+str(index_year)+'.csv'                
    #FILINGS_OUTPUT = os.path.join( CURRENT_DIRECTORY, filings_csv)
    #df_filings.to_csv(FILINGS_OUTPUT, index=False)
    
    
    #remove filings not found due to version unsupported error
    #df_filings = df_filings[df_filings['name'].notnull()]
    print_txt='Importing a total of %d/%d XML files for %s efilers'%(efiles_found,cngrants_orgs, index_year )
    print_log(LOG_FILE, print_txt,1)
    
    try:
        df_filings_cngrants = pd.merge(df_cngrants_efilers,
                 df_filings,
                 on=['OBJECT_ID','ORG_ID'])
        
    except KeyError:
        print_txt='No filings found for CN_GRANTS organizations for %s efilers'%(index_year)
        print_log(LOG_FILE, print_txt,1)
        return None         
    #remove duplicate columns
    df_filings_cngrants = df_filings_cngrants.drop( columns=['ein', 'dba'],axis=1)                 

    return df_filings_cngrants

def get_propublica_data(v_index_year, df_cngrants, LOG_FILE):
    #https://projects.propublica.org/nonprofits/api/v2/organizations/142007220.json
    baseURL = "https://projects.propublica.org/nonprofits/api/v2/organizations/"
        
    data_extracted = 0
    processed = 0
    filings = []
    cngrants_orgs = len(df_cngrants)
    index_year = int(v_index_year)
    for index, rows in df_cngrants.iterrows(): 
        ein = rows['EIN']
        queryURL = baseURL + ein+'.json'
        processed += 1
        if (processed%200 == 0):
            print('Processed %s/%s orgs, found total %s filings...'%(processed,cngrants_orgs,data_extracted))                
        #print(f"EIN to be processed is {ein}")
        #print(f"Query url is {queryURL}")
        if (requests.head(queryURL).status_code == 200):
            try:
                response = requests.get(queryURL)

                #response.text = response.text.replace('\n', '')
                org_data = response.json().get("organization", {})
                #org_data_raw = response.json()
                #org_data = org_data_raw['organization']
                #print(org_data)
                #exit(0)
                ##org_data = requests.get(queryURL).json().get("organization", [])
                
            #print(org_data)
            except Exception as e:
                print_txt='Error getting Organization data for %s\n'%ein
                print_txt=print_txt + 'Exception message is: %s'%str(e)
                print_log(LOG_FILE, print_txt,2)
                #print(print_txt)
                continue
            
            try:
                filings_data = requests.get(queryURL).json().get("filings_with_data", [])
            except Exception as e:
                print_txt='Error getting Filings with data for %s\n'%ein
                print_txt=print_txt + 'Exception message is: %s'%str(e)
                #print(print_txt)
                print_log(LOG_FILE, print_txt,2)
                continue
            #limit to last 3 years
            #filings_data = filings_data[filings_data['tax_prd_yr'] > '2017']
            #for i in range(len(filings_data)):
            
            #filings_data = requests.get(queryURL).json().get("filings_with_data", [])
            i = 0
            #try: 
            #    rec = filings_data[0]
            #except Exception as e:
            if len(filings_data) == 0:
                #print('Exception 1st element index out of range %s'%str(e))
                print_txt='Filing has no data for EIN: %s'%ein
                print_log(LOG_FILE, print_txt,2)
                continue;    
            #check if there's data for the year requested, index year being filing year
            #Fiscal year/tax period year would be filing year - 1
            while filings_data[i]['tax_prd_yr'] >= index_year-1:
                #print(f"Filing tax period for {i} is {filings_data[i]['tax_prd']}")
                filings.append({**org_data, **filings_data[i]})
                data_extracted += 1
                i += 1
                try:
                    #check if there's data to loop through, break if none
                    rec = filings_data[i]
                    #pass
                except:
                    break
                
                    #break
                    #exit(0)
        else:
            print_txt=f"No IRS990 data found for EIN {ein}."
            print_log(LOG_FILE, print_txt,2)
            continue


    print_txt='Found a total of %d/%d CNCS Orgs filings in Propublica'%(data_extracted, cngrants_orgs )
    print_log(LOG_FILE, print_txt,1)
    
    df_filings = pd.DataFrame(filings)
    #print(df_filings.columns)
    return df_filings

def print_log(LOG_FILE, print_txt, target): 
    if target == 1:
        print(print_txt) 
    with open(LOG_FILE, 'a') as f:
        f.write(print_txt)
        f.write('\n')

def main(v_db_env, v_data_name,v_index_year):

    
    VERSION_CODE = 22
    VERSION_CONFIG = 22
    CURRENT_DIRECTORY = os.getcwd()
    #today = date.today().strftime("%m_%d_%Y")
    today = datetime.now().strftime("%m_%d_%Y_%H_%M")
 
 
    data_file_log = v_data_name+'_log_'+today+'.txt'
    LOG_FILE = os.path.join(CURRENT_DIRECTORY, 'Logs',data_file_log) 
    print_txt='Data load script version: %s\n'%(VERSION_CODE)
    print_txt=print_txt + 'Config file version: %s'%(VERSION_CONFIG)    
    print_log(LOG_FILE, print_txt,1)

    start = time.time()
    print_txt='IRS990 filing year %s data import start time: %s'%(v_index_year,datetime.now().strftime("%m_%d_%Y %H:%M:%S"))
    print_log(LOG_FILE, print_txt,1)
    #Load configuration for IRS990            
    #CONFIG_FILE = os.path.join(CURRENT_DIRECTORY, 'Config','Data_file_structs.xls') 
    CONFIG_FILE = os.path.join(CURRENT_DIRECTORY, 'Config','Data_IRS_structs_v'+str(VERSION_CONFIG)+'.xls') 
    if not os.path.exists(CONFIG_FILE):
        print_txt='Error: Config file was not found: %s '%(CONFIG_FILE)
        print_log(LOG_FILE, print_txt,1)
        sys.exit(1)
    else:
        print_txt = "Config file was found: %s" %(CONFIG_FILE) 
        print_log(LOG_FILE, print_txt,1)
        

    data_name_curr = v_data_name #'IRS990'

    config_headers = pd.read_excel(CONFIG_FILE, names=['FIELDS','COLUMNS990','COLUMNS990EZ','COLUMNS990PF'],sheet_name = data_name_curr,header=None, index_col=False)
    filing_headers = config_headers['FIELDS'].T.tolist()
    filing_columns_990 = config_headers['COLUMNS990'].T.tolist()
    filing_columns_990EZ = config_headers['COLUMNS990EZ'].T.tolist()
    filing_columns_990PF = config_headers['COLUMNS990PF'].T.tolist()
    #print(filing_columns_990)
    #exit(0)
    dummy_cols = {'COLUMNS990':['careofname','address'],
                  'COLUMNS990EZ': ['careofname','address','city','state'],
                  'COLUMNS990PF': ['careofname','address','city','state']
                  }
    
    #Pull Org id and EIN from CN_GRANTS for extraction of IRS990 filings
    df_cngrants = Get_cngrants_data(v_db_env, LOG_FILE)
    #Add leading zeros, EINs are 9 digits
    df_cngrants['EIN'] = [ str(x).zfill(9) for x in df_cngrants['EIN']]    

    #ctx_auth = AuthenticationContext(SITE_URL)
    DATA_DIRECTORY = os.path.join(CURRENT_DIRECTORY, 'Data') 
    #v18. New data source Propublica to supplement IRSGOV data, limited number of fields available.
    if (data_name_curr == 'PROPUBLICA'):
        df_filings = get_propublica_data(v_index_year, df_cngrants, LOG_FILE)

        #filings_csv = 'Propublica_filings_DEBUG'+today+'.csv'                
        #FILINGS_OUTPUT = os.path.join( DATA_DIRECTORY, filings_csv)
        #df_filings.to_csv(FILINGS_OUTPUT, index=False)
        
        #Check if there're any 990 filetypes
        try:
            #if not df_filings[df_filings['formtype']==0].empty:
            df_filings_990 = df_filings[df_filings['formtype']==0][filing_columns_990]
            df_filings_990.loc[:,dummy_cols['COLUMNS990']] = np.nan
            df_filings_990.columns = filing_headers
        except KeyError as err:
            print_txt = "No 990 filings found for %s" %(v_index_year) 
            print_log(LOG_FILE, print_txt,1)

        
        #Check if there're any 990EZ filetypes        
        try:
            #df_filings[df_filings['formtype']==1].empty:
            df_filings_990EZ = df_filings[df_filings['formtype']==1][filing_columns_990EZ]
            df_filings_990EZ.loc[:,dummy_cols['COLUMNS990EZ']] = np.nan
            df_filings_990EZ.columns = filing_headers
            df_filings_990 = pd.concat([df_filings_990, df_filings_990EZ],axis=0)
        except KeyError as err:
            print_txt = "No 990EZ filings found for %s" %(v_index_year) 
            print_log(LOG_FILE, print_txt,1)

        #Check if there're any 990PF filetypes            
        try:
            df_filings_990PF = df_filings[df_filings['formtype']==2][filing_columns_990PF]
            df_filings_990PF.loc[:,dummy_cols['COLUMNS990PF']] = np.nan
            df_filings_990PF.columns = filing_headers
            df_filings_990 = pd.concat([df_filings_990, df_filings_990PF],axis=0)
        except KeyError as err:
            print_txt = "No 990PF filings found for %s" %(v_index_year) 
            print_log(LOG_FILE, print_txt,1)
            
        #Reset selected columns
        df_filings = df_filings_990

        filings_csv = 'Propublica_filings_'+today+'.csv'                
        FILINGS_OUTPUT = os.path.join( DATA_DIRECTORY, filings_csv)
        df_filings.to_csv(FILINGS_OUTPUT, index=False)
        #exit(0)
    #v18. Process only if data source is IRSGOV
    else:
        df_filings = get_irsgov_data(v_index_year, df_cngrants, LOG_FILE)

    #exit(0)
        

    #import irs990 filings into RA_IRS990 table
    tables_map = {
                    'IRSGOV': 'RA_IRS990_IRSGOV',               
                    'PROPUBLICA':'RA_IRS990_PROPUBLICA'
                    }    
    table_name = tables_map[data_name_curr]
    #print('The test table is: %s'%table_name)
    Import_filings(v_db_env, table_name,df_filings, filing_headers,LOG_FILE)
    runtime = (time.time() - start)/60
    completion = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
    print_txt='IRS990 filing year %s data extract & import completion: %s\n'%(v_index_year ,completion)
    print_txt=print_txt + 'IRS990 data process duration: %d mins'%runtime
    print_log(LOG_FILE, print_txt,1)
    return    
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Error: Invalid job execution format.\n'+
                    'Please using following format to execute the IRS990 script.\n'+
                    'Python IRS990_Process.py [Environment] [Data Source] [Filing Year].\n'+
                            'Example: Python IRS990_Process.py Test IRSGOV 2021')
        sys.exit(1)
        #raise ValueError('FAIL -- Please provide the filing year to process e.g. 2019.')
        #return
    v_db_env = sys.argv[1]
    #v_zip_file = sys.argv[3]
    v_data_name = sys.argv[2]
    v_index_year = sys.argv[3]
    
    #v_credentials = sys.argv[2]
    if (v_db_env.lower() not in ['test','prod']):
        print('Error: Invalid environment.\n'+
                    'Please use either Test/Prod for environment.\n'+ 
                            'Python IRS990_Process.py [Environment] [Data Source] [Filing Year].\n'+
                            'Example: Python IRS990_Process.py Test IRSGOV 2021')
        sys.exit(1)
    if (v_data_name.upper() not in ['IRSGOV','PROPUBLICA']):
        print('Error: Invalid DATA SOURCE.\n'+
                    'Please use either IRSGOV or PROPUBLICA for Data Source.\n'+ 
                            'Python IRS990_Process.py [Environment] [Data Source] [Filing Year].\n'+
                            'Example: Python IRS990_Process.py Test IRSGOV 2021')
        sys.exit(1)

    try:
        v_index_year = int(sys.argv[3])
    except:
        print('Error: Filing year needs to be valid numeric year.\n'+
                    'Please use filing years later than 2016.\n'+ 
                            'Python IRS990_Process.py [Environment] [Data Source] [Filing Year].\n'+
                            'Example: Python IRS990_Process.py Test IRSGOV 2021')
        sys.exit(1)   
    
    #if (v_index_year>datetime.now().year or v_index_year<2016):
    if (v_index_year<2017):
        #print('Please use filing years later than 2016. current calendar year: %d.'%(datetime.now().year))        
        print('Please use filing years later than 2016')        
        sys.exit(1)
    
    #index_year = '2021'
    main(v_db_env, v_data_name, v_index_year)