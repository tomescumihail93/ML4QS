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

# Let is create our visualization class again.
DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles/'
dataset = pd.read_csv(dataset_path + 'chapter3_result_outliers.csv', index_col=0)
dataset.index = dataset.index.to_datetime()

# Computer the number of milliseconds covered by an instane based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

# Step 2: Let us impute the missing values.

MisVal = ImputationMissingValues()
imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'hr_watch_rate')
imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'hr_watch_rate')
imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'hr_watch_rate')
DataViz.plot_imputed_values(dataset, ['original', 'mean', 'interpolation'], 'hr_watch_rate', imputed_mean_dataset['hr_watch_rate'], imputed_interpolation_dataset['hr_watch_rate'])

# And we impute for all columns except for the label in the selected way (interpolation)

for col in [c for c in dataset.columns if not 'label' in c]:
    dataset = MisVal.impute_interpolate(dataset, col)

# Let us try the Kalman filter on the heart rate

original_dataset = pd.read_csv(dataset_path + 'cs_chapter2_result.csv', index_col=0)
original_dataset.index = original_dataset.index.to_datetime()
KalFilter = KalmanFilters()
kalman_dataset = KalFilter.apply_kalman_filter(original_dataset, 'hr_watch_rate')
DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], 'hr_watch_rate', kalman_dataset['hr_watch_rate_kalman'])
DataViz.plot_dataset(kalman_dataset, ['hr_watch_rate', 'hr_watch_rate_kalman'], ['exact','exact'], ['line', 'line'])

dataset.to_csv(dataset_path + 'chapter3_result_final.csv')