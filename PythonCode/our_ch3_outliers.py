##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import copy
import pandas as pd
import numpy as np

# Let is create our visualization class again.
DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sture the index is of the type datetime.
dataset_path = './intermediate_datafiles/'
try:
    dataset_own = pd.read_csv(dataset_path + 'chapter2_result_own.csv', index_col=0)
    dataset_cs = pd.read_csv(dataset_path + 'chapter2_result_cs.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset_own.index = dataset_own.index.to_datetime()
dataset_cs.index = dataset_cs.index.to_datetime()

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = (dataset_own.index[1] - dataset_own.index[0]).microseconds/1000

# Step 1: Let us see whether we have some outliers we would prefer to remove.

# Determine the columns we want to experiment on.
outlier_columns = ['acc_phone_x','acc_phone_y']

# Create the outlier classes.
OutlierDistr = DistributionBasedOutlierDetection()
OutlierDist = DistanceBasedOutlierDetection()

# Parameters that can be played around with for different outlier detection methods
# Chauvenet
constant = 2 # given was 2
# Mixture models
NumDist = 3 # given was 3
# Simple Distance
dmin = 0.10 # given was 0.10
fmin = 0.99 # given was 0.99
# Local outlier factor
k = 5 # given was 5

##### Outlier filtering for the CS dataset #####

#And investigate the approaches for all relevant attributes.
for col in outlier_columns:
    # And try out all different approaches. Note that we have done some optimization
    # of the parameter values for each of the approaches by visual inspection.
    dataset_cs = OutlierDistr.chauvenet(dataset_cs, col, constant)
    DataViz.plot_binary_outliers(dataset_cs, col, col + '_outlier')
    dataset_cs = OutlierDistr.mixture_model(dataset_cs, col, NumDist)
    DataViz.plot_dataset(dataset_cs, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])
    # This requires:
    # n_data_points * n_data_points * point_size =
    # 31839 * 31839 * 64 bits = ~8GB available memory
    try:
        dataset_cs = OutlierDist.simple_distance_based(dataset_cs, [col], 'euclidean', dmin, fmin)
        DataViz.plot_binary_outliers(dataset_cs, col, 'simple_dist_outlier')
    except MemoryError as e:
        print('Not enough memory available for simple distance-based outlier detection...')
        print('Skipping.')
    
    try:
        dataset_cs = OutlierDist.local_outlier_factor(dataset_cs, [col], 'euclidean', k)
        DataViz.plot_dataset(dataset_cs, [col, 'lof'], ['exact','exact'], ['line', 'points'])
    except MemoryError as e:
        print('Not enough memory available for lof...')
        print('Skipping.')

    # Remove all the stuff from the dataset again.
    cols_to_remove = [col + '_outlier', col + '_mixture', 'simple_dist_outlier', 'lof']
    for to_remove in cols_to_remove:
        if to_remove in dataset_cs:
            del dataset_cs[to_remove]

# We take Chauvent's criterion and apply it to all but the label data...

for col in [c for c in dataset_cs.columns if not 'label' in c]:
    print('Measurement is now: ' , col)
    dataset_cs = OutlierDistr.chauvenet(dataset_cs, col, constant)
    dataset_cs.loc[dataset_cs[col + '_outlier'] == True, col] = np.nan
    del dataset_cs[col + '_outlier']

dataset_cs.to_csv(dataset_path + 'chapter3_result_outliers_cs.csv')

##### Outlier filtering for our own dataset #####

#And investigate the approaches for all relevant attributes.
for col in outlier_columns:
    # And try out all different approaches. Note that we have done some optimization
    # of the parameter values for each of the approaches by visual inspection.
    dataset_own = OutlierDistr.chauvenet(dataset_own, col, constant)
    DataViz.plot_binary_outliers(dataset_own, col, col + '_outlier')
    dataset_own = OutlierDistr.mixture_model(dataset_own, col, NumDist)
    DataViz.plot_dataset(dataset_own, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])
    # This requires:
    # n_data_points * n_data_points * point_size =
    # 31839 * 31839 * 64 bits = ~8GB available memory
    try:
        dataset_own = OutlierDist.simple_distance_based(dataset_own, [col], 'euclidean', dmin, fmin)
        DataViz.plot_binary_outliers(dataset_own, col, 'simple_dist_outlier')
    except MemoryError as e:
        print('Not enough memory available for simple distance-based outlier detection...')
        print('Skipping.')
    
    try:
        dataset_own = OutlierDist.local_outlier_factor(dataset_own, [col], 'euclidean', k)
        DataViz.plot_dataset(dataset_own, [col, 'lof'], ['exact','exact'], ['line', 'points'])
    except MemoryError as e:
        print('Not enough memory available for lof...')
        print('Skipping.')

    # Remove all the stuff from the dataset again.
    cols_to_remove = [col + '_outlier', col + '_mixture', 'simple_dist_outlier', 'lof']
    for to_remove in cols_to_remove:
        if to_remove in dataset_own:
            del dataset_own[to_remove]

# We take Chauvent's criterion and apply it to all but the label data...

for col in [c for c in dataset_own.columns if not 'label' in c]:
    print('Measurement is now: ' , col)
    dataset_own = OutlierDistr.chauvenet(dataset_own, col, constant)
    dataset_own.loc[dataset_own[col + '_outlier'] == True, col] = np.nan
    del dataset_own[col + '_outlier']

dataset_own.to_csv(dataset_path + 'chapter3_result_outliers_own.csv')