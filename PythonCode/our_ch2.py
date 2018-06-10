##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Set paths
dataset_path_own = '../our_dataset/'
dataset_path_cs = '../crowdsignals_dataset/'
result_dataset_path = './intermediate_datafiles/'

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
import copy
import os

# Create folder if needed
if not os.path.exists(result_dataset_path):
    print('Creating result directory: ' + result_dataset_path)
    os.makedirs(result_dataset_path)

# Chapter 2: Initial exploration of the dataset.

# Set a granularity (i.e. how big are our discrete time steps). We start very
# coarse grained, namely one measurement per minute, and secondly use four measurements
# per second

granularities = [60000, 250]
datasets_own = []
datasets_cs = []

for milliseconds_per_instance in granularities:

    # Create an initial dataset object with the base directory for our data and a granularity
    DataSetOwn = CreateDataset(dataset_path_own, milliseconds_per_instance)
    DataSetCS = CreateDataset(dataset_path_cs, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values/
    DataSetOwn.add_numerical_dataset('acc_custom.csv', 'timestamps', ['x','y','z'], 'avg', 'acc_phone_')
    DataSetCS.add_numerical_dataset('accelerometer_phone.csv', 'timestamps', ['x','y','z'], 'avg', 'acc_phone_')
    DataSetCS.add_numerical_dataset('accelerometer_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'acc_watch_')

    # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values/
    DataSetOwn.add_numerical_dataset('gyro_custom.csv', 'timestamps', ['x','y','z'], 'avg', 'gyr_phone_')
    DataSetCS.add_numerical_dataset('gyroscope_phone.csv', 'timestamps', ['x','y','z'], 'avg', 'gyr_phone_')
    DataSetCS.add_numerical_dataset('gyroscope_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'gyr_watch_')

    # We add the heart rate (continuous numerical measurements) and aggregate by averaging again
    DataSetCS.add_numerical_dataset('heart_rate_smartwatch.csv', 'timestamps', ['rate'], 'avg', 'hr_watch_')

    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).
    DataSetOwn.add_event_dataset('labels_custom.csv', 'label_start', 'label_end', 'label', 'binary')
    DataSetCS.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', 'binary')

    # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging again
    # amount of light sensed is missing from our dataset
    DataSetCS.add_numerical_dataset('light_phone.csv', 'timestamps', ['lux'], 'avg', 'light_phone_')

    # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    DataSetOwn.add_numerical_dataset('mag_custom.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_phone_')
    DataSetOwn.add_numerical_dataset('press_custom.csv', 'timestamps', ['pressure'], 'avg', 'press_phone_')
    DataSetCS.add_numerical_dataset('magnetometer_phone.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_phone_')
    DataSetCS.add_numerical_dataset('magnetometer_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_watch_')

    # We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again
    DataSetOwn.add_numerical_dataset('pedom_custom.csv', 'timestamps', ['steps', 'distance'], 'avg', 'pedom_phone_')
    DataSetCS.add_numerical_dataset('pressure_phone.csv', 'timestamps', ['pressure'], 'avg', 'press_phone_')

    # Get the resulting pandas data table

    dataset_own = DataSetOwn.data_table
    dataset_cs = DataSetCS.data_table

    # Plot the data
    DataViz = VisualizeDataset()

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset_own, ['acc_phone_x','acc_phone_y','acc_phone_z'])
    DataViz.plot_dataset_boxplot(dataset_cs, ['acc_phone_x','acc_phone_y','acc_phone_z'])

    # Plot all data
    DataViz.plot_dataset(dataset_own, ['acc_', 'gyr_', 'mag_', 'press_' ,'pedom_phone_', 'label'], ['like', 'like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line','line', 'points', 'points'])
    DataViz.plot_dataset(dataset_cs, ['acc_phone', 'gyr_phone', 'mag_phone', 'press_phone_', 'label'], ['like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'])

    # And print a summary of the dataset

    util.print_statistics(dataset_own)
    datasets_own.append(copy.deepcopy(dataset_own))
    
    util.print_statistics(dataset_cs)
    datasets_cs.append(copy.deepcopy(dataset_cs))

# And print the table that has been included in the book

util.print_latex_table_statistics_two_datasets(datasets_own[0], datasets_own[1])
util.print_latex_table_statistics_two_datasets(datasets_cs[0], datasets_cs[1])

# Finally, store the last dataset we have generated (250 ms).
dataset_own.to_csv(result_dataset_path + 'chapter2_result_own.csv')
dataset_cs.to_csv(result_dataset_path + 'chapter2_result_cs.csv')
