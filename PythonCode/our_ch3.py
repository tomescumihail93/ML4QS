##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

# Let us create our visualization class again.
DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles/'
dataset_own = pd.read_csv(dataset_path + 'chapter3_result_outliers_own.csv', index_col=0)
dataset_cs = pd.read_csv(dataset_path + 'chapter3_result_outliers_cs.csv', index_col=0)

# Create index
dataset_own.index = dataset_own.index.to_datetime()
dataset_cs.index = dataset_cs.index.to_datetime()

# Computer the number of milliseconds covered by an instane based on the first two rows
milliseconds_per_instance = (dataset_own.index[1] - dataset_own.index[0]).microseconds/1000

# Step 2: Let us impute the missing values for CS (plot only).
MisVal = ImputationMissingValues()
imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset_cs), 'hr_watch_rate')
imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset_cs), 'hr_watch_rate')
imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset_cs), 'hr_watch_rate')
DataViz.plot_imputed_values(dataset_cs, ['original', 'mean', 'interpolation'], 'hr_watch_rate', imputed_mean_dataset['hr_watch_rate'], imputed_interpolation_dataset['hr_watch_rate'])

# Make plot for own dataset
MisVal = ImputationMissingValues()
imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset_own), 'press_phone_pressure')
imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset_own), 'press_phone_pressure')
imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset_own), 'press_phone_pressure')
DataViz.plot_imputed_values(dataset_own, ['original', 'mean', 'interpolation'], 'press_phone_pressure', imputed_mean_dataset['press_phone_pressure'], imputed_interpolation_dataset['press_phone_pressure'])

# Imputation for the CS dataset
for col in [c for c in dataset_cs.columns if not 'label' in c]:
    dataset_cs = MisVal.impute_interpolate(dataset_cs, col)

dataset_cs.to_csv(dataset_path + 'chapter3_result_final_cs.csv')

# Imputation for our dataset
for col in [c for c in dataset_own.columns if not 'label' in c]:
    dataset_own = MisVal.impute_interpolate(dataset_own, col)

dataset_own.to_csv(dataset_path + 'chapter3_result_final_own.csv')

# Let us try the Kalman filter on the heart rate
#original_dataset = pd.read_csv(dataset_path + 'chapter3_result_outliers_cs.csv', index_col=0)
#original_dataset.index = original_dataset.index.to_datetime()
#KalFilter = KalmanFilters()
#kalman_dataset = KalFilter.apply_kalman_filter(original_dataset, 'hr_watch_rate')
#DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], 'hr_watch_rate', kalman_dataset['hr_watch_rate_kalman'])
#DataViz.plot_dataset(kalman_dataset, ['hr_watch_rate', 'hr_watch_rate_kalman'], ['exact','exact'], ['line', 'line'])
