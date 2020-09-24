# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:38:39 2020
Author: Audrey Zhang
NOTE: run initial_exploration.py for base datasets first
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from itertools import cycle, islice
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#%%

""" first create custering_task function using kmeans"""

def clustering_task(D, N):
    ### BEGIN SOLUTION
    kmeans = KMeans(n_clusters=N)
    model = kmeans.fit(D)
    clusters=pd.DataFrame(model.cluster_centers_)
    return clusters

#%%

""" create support functions for visualizations later"""

def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P


# Function that creates Parallel Plots

def parallel_plot(data):

    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))

    plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])

    parallel_coordinates(data, 'prediction', color = my_colors, marker='o')


#%%

""" start clusting analysis for outcomes, based on total sales and loyalty status """

features=['pct_active', 'TOT_NET_SLS_AMT', 'AER_NET_SLS_AMT', 'isCust_Loyalty', 'isCust_AECredit']
#%%
key_outcomes=summary_data[features].copy()

# there were a few rows with 0 in total or AER net sales columns, drop for analysis

key_outcomes=key_outcomes[(key_outcomes['TOT_NET_SLS_AMT']>0) & (key_outcomes['AER_NET_SLS_AMT']>0)]

# scale data since purchase amounts vary quite a bit 
scaled_data = StandardScaler().fit_transform(key_outcomes)

#%%

""" use Within Cluster Sum of Squared Errors (WSS) to find optimal number of clusters """

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    
# plot to show outcomes 

plt.figure(figsize=(12,6))    
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

#%%

# based on wcss analysis, the optimal number of clusters is 5

outcome_clusters=KMeans(n_clusters=5)
test_m=outcome_clusters.fit(scaled_data)


centers=outcome_clusters.cluster_centers_

#%%
    
p_outcomes = pd_centers(features, centers)
parallel_plot(p_outcomes)

#%%

""" use similar method to run analysis for initial customer buckets""" 

init_data=pd.read_csv(r"C:\Users\audre\OneDrive\Documents\CMU\Fall_2020\AEO_Case\comp_data\AEO_comp_D1_copy.csv")

#%%

# note that dummy vars were created initial_exploration.py
acquis_data=pd.concat([init_data, brand_dummy, store_dummy, summary_data['channel'], customer_activity[]])

acquis_data=acquis_data.drop(['MASKED_CUST_NBR', 'AER_acquisition_channel', 'AER_acq_dt', 'STORE_BRAND_CD', 'STORE_FORMAT_CD', 'MALL_TYP_DESC', 'CAMPAIGN_CATEGORY', 'CAMPAIGN_CHANNEL', 'CAMPAIGN_PARTNER_NBR'], axis=1)

#%%
# fill in NaN values with 0 
acquis_data=acquis_data.fillna(0)

#%%
""" there are too many variables; use feature selection/dimensionality reduction to identify key features """

# first get dataset with numeric variables only
num_vars=acquis_data.loc[:, acquis_data.max(axis=0)>1]
#%%
# then get variance sorted

print(num_vars.var().sort_values(ascending=True))

# based on output, the various "sweater_itm_qty" and similar measures have very small variance (<1). 
# others ike accessories_net_sls and offline_bras_net_sls have var less than 6.
# aer_itm_qty and tot_itm_qty have variances of 6.3 and 8.5, respectively. 
# Since total and aer item qty purchases will likely be important as an aggregate measure, 
# the variance based feature reduction will be set at a limit of <6.

#%%
variables_to_drop=[]
var_list=num_vars.var()
for i in range(0, len(var_list)):
    if var_list[i]<6:
        variables_to_drop.append(var_list.index[i])

acquis_data=acquis_data.drop(variables_to_drop, axis=1)

#%%

rf_model=RandomForestRegressor(random_state=1, max_depth=10)
#model.fit(acquis_data)
