# -*- coding: utf-8 -*-
"""
date: September 18, 2020
author: Audrey Zhang
title: initial_exploration.py
description: initial exploration of AEO case competition data, 
    and creation of various summary datasets for analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

aeo_data=pd.read_csv(r"C:\Users\audre\OneDrive\Documents\CMU\Fall_2020\AEO_Case\comp_data\AEO_combined_copy.csv")

#%%

# set up month index from 1 through 24
month_idx=np.array(np.arange(1,25))

"""
Create monthly sales data, including the total and average cumlative net sales for all AE+Aerie sales
and for Aerie only sales by month # after acquisition. 

Note that later months should be interpreted with caution as data may not be complete for newer customers

"""
mon_sales=np.zeros([24, 3])

mon_sls_vars=[col for col in list(aeo_data.columns) if 'cum' in col and ('_NET_SLS_AMT' in col or '_ITM_QTY' in col) and '_p' not in col]

for c in range(3):
    for r in range(24):
        mon=r+1
        if mon<10:
            mon='0'+str(mon)
        else: 
            mon=str(mon)
        var=mon_sls_vars[c]+'_p'+mon
        mon_sales[r][c]=np.around(aeo_data[var].sum(skipna=True), decimals=2)

mon_sales_data=pd.DataFrame(mon_sales, index=month_idx, columns= mon_sls_vars)

for r in range(24):
    mon=r+1
    if mon<10:
        mon='0'+str(mon)
    else: 
        mon=str(mon)
    var='isActive_p'+mon
    mon_sales_data.at[r+1, 'num_active']=aeo_data[var].sum()
    
for c in range(len(mon_sls_vars)):
    var=mon_sls_vars[c]
    new_var='ave_'+var
    mon_sales_data[new_var]=mon_sales_data[var]/mon_sales_data['num_active']

#%%

# create pct_AER variable, which is the % of total AE + Aerie sales that were Aerie sales 

mon_sales_data['pct_AER']=mon_sales_data['ave_cum_AER_NET_SLS_AMT']/mon_sales_data['ave_cum_NET_SLS_AMT']*100


#%%

""" 
identify customer retension rate and trends
"""

customer_activity=aeo_data[['MASKED_CUST_NBR'] + [col for col in list(aeo_data.columns) if 'isActive' in col]].copy()
#customer_activity.set_index(aeo_data['MASKED_CUST_NBR'])

#%%
customer_activity['tot_mo']=customer_activity.count(axis=1)

customer_activity['tot_active_mo']=customer_activity[['isActive_p%s' % str(i).zfill(2) for i in range(1,25)]].sum(axis=1)

customer_activity['pct_active_mo']=customer_activity['tot_active_mo']/customer_activity['tot_mo']

#%%
"""
create summary table of activity by month
"""

activity=np.zeros([24, 2])

for i in range(24):
    mo=str(i+1).zfill(2)
    activity[i,0]=customer_activity['isActive_p%s' % mo].count()
    activity[i,1]=customer_activity['isActive_p%s' % mo].sum()

activity_monthly=pd.DataFrame(activity, index=month_idx, columns=['num_all_customers', 'num_active_customers'])

#%%

"""
calculate returning customer rate
"""
print(customer_activity[customer_activity['tot_active_mo']>1].shape[0]/customer_activity.shape[0])


#%% 
"""
Calculate correlation matrix to explore relationship between summary vars
"""

summary_data=aeo_data[[col for col in list(aeo_data.columns) if '_p' not in col and '_QTY' not in col]].copy()

summary_data=summary_data.merge(customer_activity[['MASKED_CUST_NBR', 'tot_mo', 'tot_active_mo']])

#%% 
summary_data['channel']=0

summary_data[summary_data['AER_acquisition_channel']=='ONLINE']['channel']=1

brand_dummy=pd.get_dummies(summary_data['STORE_BRAND_CD'])

store_dummy=pd.get_dummies(summary_data['STORE_FORMAT_CD'])

summary_data=pd.concat([summary_data, brand_dummy, store_dummy], axis=1)
#%%

summary_data=summary_data.drop(['MASKED_CUST_NBR', 'CAMPAIGN_PARTNER_NBR', 'AER_acq_dt', 'AER_acquisition_channel', 'STORE_BRAND_CD', 'STORE_FORMAT_CD', 'MALL_TYP_DESC', 'CAMPAIGN_CATEGORY', 'CAMPAIGN_CHANNEL'], axis=1)


 #%%
 
 