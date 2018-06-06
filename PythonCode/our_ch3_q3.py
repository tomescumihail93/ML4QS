# Chapter 3 Exercise 3

# import packages, classes etc.
import pandas as pd
#from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
#import numpy as np
#from pykalman import KalmanFilter

# Load in data from CrowdSignals
Filename = '..\crowdsignals_dataset\heart_rate_smartwatch.csv'
Data = pd.read_csv(Filename)

# Get list with variable names of Dataset
ListVars = list(Data.columns.values)

# Create Dataframe with only Heart rates and check how many NaNs
#HeartRate = Data['rate']
#NaNs = HeartRate.isnull().sum()

# Imputation of Heart Rate via Kalman Filter
HRFilter = KalmanFilters()
FilteredHeartRate = HRFilter.apply_kalman_filter(Data,'rate')

# Comparison between data and impute heart rate and print first 10 obs vs imputations
RealVSKalman = FilteredHeartRate[['rate','rate_kalman']]
print(RealVSKalman.head(10))
# maybe calculate some statistics..
#...


