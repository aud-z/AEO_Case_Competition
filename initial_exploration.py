# -*- coding: utf-8 -*-
"""
date: September 18, 2020
author: Audrey Zhang
title: initial_exploration.py
description: initial explo"ration of AEO case competition data, 
    and creation of various summary datasets for analysis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from itertools import cycle, islice
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


#%%
aeo_data=pd.read_csv(r"C:\Users\audre\OneDrive\Documents\CMU\Fall_2020\AEO_Case\comp_data\AEO_combined_copy.csv")



#%%

# take a look at where customers are coming from, and how many are new 

print(pd.crosstab(aeo_data.isAE_migration, aeo_data.AER_acquisition_channel, normalize='columns'))


#%%

# analyze net sales by isAE_migration category
print(aeo_data[['isAE_migration', 'cum_NET_SLS_AMT']].groupby('isAE_migration').describe().unstack(1))


#%%

# analyze net sales by campaign channel
print('cumulative aerie net sales by campaign category: ')
print(aeo_data[['CAMPAIGN_CATEGORY', 'cum_AER_NET_SLS_AMT']].groupby('CAMPAIGN_CATEGORY').sum())

#%%
print('get counts of new customers by campaign channel')
print(pd.crosstab(aeo_data.isAE_migration, aeo_data.CAMPAIGN_CATEGORY).unstack(1))


#%%

# next, take a look at the AE credit card sales

print(aeo_data[aeo_data['isCust_AECredit']==1]['cum_AER_NET_SLS_AMT'].sum()/aeo_data['cum_AER_NET_SLS_AMT'].sum())

print(aeo_data[aeo_data['isCust_AECredit']==1]['cum_NET_SLS_AMT'].sum()/aeo_data['cum_NET_SLS_AMT'].sum())


#%%

# set up month index from 1 through 24
month_idx=np.array(np.arange(1,25))

#%%

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

customer_activity=aeo_data[['MASKED_CUST_NBR', 'cum_NET_SLS_AMT', 'cum_AER_NET_SLS_AMT', 'isCust_AECredit','isCust_AECredit_preAcq'] + [col for col in list(aeo_data.columns) if 'isActive' in col]].copy()

#%%

active_vars=list(customer_activity[['isActive_p%s' % str(i).zfill(2) for i in range(1,25)]].columns)

#%%
customer_activity['tot_mo']=customer_activity[active_vars].count(axis=1)

#%%
customer_activity['tot_active_mo']=customer_activity[active_vars].sum(axis=1)

customer_activity['pct_active_mo']=customer_activity['tot_active_mo']/customer_activity['tot_mo']

#%%

""" 
create some helper functions and variables to estimate customer value, base on an RMF score - recency, monetary, frequency 
"""
customer_activity['latest_active_mo']=0

#%%

for i in range(24, 0, -1):
    customer_activity.loc[(customer_activity['latest_active_mo']==0) & (customer_activity['isActive_p%s' % str(i).zfill(2)]==1), 'latest_active_mo']=i

#%%

# create variable for recency, which calculates the # of months between now and most recent purchase
# note that smaller value indicates more recent activity, with 0 being the smallest possible value 

customer_activity['recency']=customer_activity['tot_mo']-customer_activity['latest_active_mo']


#%% 

# create variable for average spend per active month 
customer_activity['cum_AER_NET_SLS_AMT']=customer_activity['cum_AER_NET_SLS_AMT'].fillna(0)
customer_activity['cum_NET_SLS_AMT']=customer_activity['cum_NET_SLS_AMT'].fillna(0)

customer_activity['ave_aer_mo_spd']=customer_activity['cum_AER_NET_SLS_AMT']/customer_activity['tot_active_mo']
customer_activity['ave_mo_spd']=customer_activity['cum_NET_SLS_AMT']/customer_activity['tot_active_mo']
#%%

customer_activity['isCust_AECredit_preAcq']=customer_activity['isCust_AECredit_preAcq'].fillna(0)
customer_activity['AE_cred']=customer_activity['isCust_AECredit_preAcq']+customer_activity['isCust_AECredit']
#%%

# create data subset to calculate adjusted RFM score 

rfm=customer_activity[['MASKED_CUST_NBR', 'pct_active_mo', 'tot_mo', 'recency', 'ave_aer_mo_spd', 'AE_cred']]
#%%

# since not all customers have been in the database for the same length of time
# assign median recency to customers who only has 1 month of data
# this way we are not skewed by recent customers who shows 0 recency in their 1st month of history

rfm['adjusted_r']=rfm['recency']
rfm.loc[rfm['tot_mo']==1, 'adjusted_r']=rfm['recency'].median()

#%%

# for pct active mo, we want to adjust the customers with only 1 total month of data 
# because they will show a pct_active rate of 100% 
# which may skew the analysis 
rfm['adjusted_f']=rfm['pct_active_mo']
rfm.loc[rfm['tot_mo']==1, 'adjusted_f']=rfm['pct_active_mo'].median()
#%%
rfm['r_quartile']=pd.qcut(rfm['adjusted_r'], 4, ['4', '3', '2', '1'])
rfm['f_quartile']=pd.qcut(rfm['adjusted_f'], 4, ['1', '2', '3', '4'])
rfm['m_quartile']=pd.qcut(rfm['ave_aer_mo_spd'], 4, ['1', '2', '3', '4'])

rfm['rfm_wtd_score']=rfm['r_quartile'].astype(str)+rfm['f_quartile'].astype(str)+rfm['m_quartile'].astype(str)+rfm['AE_cred'].astype(str)
#%%
rfm['numeric']=rfm['r_quartile'].astype(int)+rfm['f_quartile'].astype(int)+rfm['m_quartile'].astype(int)+rfm['AE_cred'].astype(int)
#%%

ax=sns.countplot(x='numeric', data=rfm)
#%%
rfm['score_quartile']=pd.qcut(rfm['numeric'], 4, ['1', '2', '3', '4'])

#%%
customer_activity.to_csv(r'C:\Users\audre\Documents\customer_activity.csv', index=False)
#rfm.to_csv(r'C:\Users\audre\Documents\rfm_scores.csv', index=False)
#%%

desirable=rfm[rfm['score_quartile']=='4'].copy()


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
get summary dataset with sales data
@AZ: reduce redundancies!!! 

"""

summary_data=aeo_data[[col for col in list(aeo_data.columns) if '_p' not in col and '_QTY' not in col]].copy()
#%%
summary_data=summary_data.merge(customer_activity[['MASKED_CUST_NBR', 'tot_mo', 'tot_active_mo', 'recency']])

#%% 

summary_data['channel']=0

summary_data['channel']=np.where(summary_data['AER_acquisition_channel']=='ONLINE', 1, 0)

brand_dummy=pd.get_dummies(summary_data['STORE_BRAND_CD'])

store_dummy=pd.get_dummies(summary_data['STORE_FORMAT_CD'])

summary_data=pd.concat([summary_data, brand_dummy, store_dummy], axis=1)
#%%

summary_data=summary_data.drop(['MASKED_CUST_NBR', 'CAMPAIGN_PARTNER_NBR', 'AER_acq_dt', 'AER_acquisition_channel', 'STORE_BRAND_CD', 'STORE_FORMAT_CD', 'MALL_TYP_DESC', 'CAMPAIGN_CATEGORY', 'CAMPAIGN_CHANNEL'], axis=1)

#%%

"""

#%%

#abbrev_data=summary_data[[col for col in list(summary_data.columns) if '_AMT' not in col and 'cum_' not in col]].copy()

#abbrev_data[[col for col in list(abbrev_data.columns)]].fillna(0, axis=1)
#abbrev_data.fillna(0)
acquis_data=pd.read_csv(r"C:\Users\audre\OneDrive\Documents\CMU\Fall_2020\AEO_Case\comp_data\AEO_comp_D1_copy.csv")
#%%
acquis_data=pd.concat([acquis_data, brand_dummy, store_dummy, summary_data['channel']])
acquis_data=acquis_data.drop(['MASKED_CUST_NBR', 'AER_acquisition_channel', 'AER_acq_dt', 'STORE_BRAND_CD', 'STORE_FORMAT_CD', 'MALL_TYP_DESC', 'CAMPAIGN_CATEGORY', 'CAMPAIGN_CHANNEL', 'CAMPAIGN_PARTNER_NBR'], axis=1)

#%%

acquis_data=acquis_data.fillna(0)
#%%
kmeans=KMeans(n_clusters=3)

kmeans.fit(acquis_data)

print(kmeans.cluster_centers_) 
#%%
y_km=kmeans.fit_predict(acquis_data)

#%%
plt.scatter(acquis_data[y_km ==0,0], points[y_km == 0,1], s=100, c='red')
plt.scatter(acquis_data[y_km ==1,0], points[y_km == 1,1], s=100, c='black')
plt.scatter(acquis_data[y_km ==2,0], points[y_km == 2,1], s=100, c='blue')
#plt.scatter(points[y_km ==3,0], points[y_km == 3,1], s=100, c='cyan')
"""

#%%

