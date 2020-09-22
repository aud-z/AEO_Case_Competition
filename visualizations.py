# -*- coding: utf-8 -*-
"""
date: September 20, 2020
author: Audrey Zhang
title: visualizations.py
description: initial exploration of AEO case competition data with visualizations
NOTE: RUN INITIAL_EXPLORATION.PY FIRST 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns

#%%

"""
plot monthly sales trendlines
for total AE+Aerie sales and Aerie sales only 

"""

fig, ax=plt.subplots()

mon_sales_data[['ave_cum_NET_SLS_AMT', 'ave_cum_AER_NET_SLS_AMT']].plot(secondary_y=True, ax=ax, label=['average net sales (cumulative)', 'average net sales Aerie (cumulative)'])

mon_sales_data['pct_AER'].plot(ax=ax, style='--g', label='% Aerie sales')

lines=ax.get_lines()+ax.right_ax.get_lines()
ax.legend(lines, ['% Aerie sales', 'avg cumulative net sales', 'avg cumulative net sales Aerie'], loc='upper left')
ax.set_ylabel('% of total sales')
ax.right_ax.set_ylabel('net sales $')
plt.xlim(1,24)
plt.ylim(0)
ax.set_xlabel('month after acquisition')
plt.title('Avg cumulative net sales in months 1-24 after acquisition')

plt.show()

#%%
"""
plot monthly activity
for total # of customers by length of month after acquisition 
and # of customers with activity in month after acquisition 
"""

activity_monthly['num_all_customers'].plot(kind='bar', stacked=True)
activity_monthly['num_active_customers'].plot(kind='bar', stacked=True, color='orange')

plt.legend(['total # of customers', '# of active customers'], loc='upper right')

plt.xlabel('# of months after acquisition')

plt.ylabel('# of customers')
plt.title('Customer activity by month after acquisition')
plt.show()


#%%

"""
create correlation matrix for key features
note that dummy variables have been constructed for various categorical vars
including: 
    
    AER_acquisition_channel'= 1 if ONLINE'
        and 0 if 'STORES'
    STORE_FORMAT_CD
    STORE_BRAND_CD
    
"""

# Compute the correlation matrix
corr = summary_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

ax.set_title('Correlation Matrix')

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": .5})

#%%

"""create median spending chart by loyalty type"""

# first create summary spending table with median spending information 
sum_spend=pd.DataFrame(np.zeros([3,3]), columns=['category', 'tot_net_sls', 'tot_ae_sls'])
#%%

# columns in order are: category, total net sales, and total AE sales
sum_spend.iloc[0, 0]='neither'
sum_spend.iloc[0, 1]=np.median(summary_data[(summary_data['isCust_Loyalty']!=1) & (summary_data['isCust_AECredit']!=1)]['TOT_NET_SLS_AMT'])
sum_spend.iloc[0, 2]=np.median(summary_data[(summary_data['isCust_Loyalty']!=1) & (summary_data['isCust_AECredit']!=1)]['AER_NET_SLS_AMT'])
sum_spend.iloc[1, 0]='loyalty'
sum_spend.iloc[1, 1]=np.median(summary_data[summary_data['isCust_Loyalty']==1]['TOT_NET_SLS_AMT'])
sum_spend.iloc[1, 2]=np.median(summary_data[summary_data['isCust_Loyalty']==1]['AER_NET_SLS_AMT'])
sum_spend.iloc[2, 0]='ae credit'
sum_spend.iloc[2, 1]=np.median(summary_data[summary_data['isCust_AECredit']==1]['TOT_NET_SLS_AMT'])
sum_spend.iloc[2, 2]=np.median(summary_data[summary_data['isCust_AECredit']==1]['AER_NET_SLS_AMT'])


#%%

fig=sum_spend.plot(x='category', y=['tot_net_sls', 'tot_ae_sls'], kind='bar')
fig.set_xlabel('loyalty program category')
plt.ylim(0, 75)
fig.set_ylabel('median spending $')
fig.legend(['total net sales', 'total AE net sales'])
fig.set_title('Median Spending by Loyalty Program')
for p in fig.patches:
    fig.annotate(str(p.get_height()), (p.get_x()*1.025, p.get_height()*1.015), fontsize=8)
plt.show()
