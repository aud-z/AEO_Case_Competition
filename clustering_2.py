# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:30:08 2020
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

#rfm_features=['r_quartile', 'm_quartile', 'f_quartile', 'AE_cred']
rfm_features=['adjusted_r', 'ave_aer_mo_spd', 'adjusted_f', 'AE_cred']

rfm_outcomes=rfm[rfm_features].copy()

rfm_scaled = StandardScaler().fit_transform(rfm_outcomes)


wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(rfm_scaled)
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

outcome_clusters=KMeans(n_clusters=5)
test_m=outcome_clusters.fit(rfm_scaled)


centers=outcome_clusters.cluster_centers_

#%%
    
p_outcomes = pd_centers(rfm_features, centers)
parallel_plot(p_outcomes)

