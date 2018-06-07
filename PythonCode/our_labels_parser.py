import json
import pandas as pd
import numpy as np

# df = pd.read_csv('../our_dataset/our_data_last.csv')

# Function that takes the date from the recorded labels
# searches for that specific date in the big dataset
# after which it takes one row, and from that row the unix timestamp
# coresponding to the date the label was registered
# 
# Example: 2018-06-07 18:13:50 -> it will look for all rows with this date, 
# take the first element, and use it's unix timestamp to create the labels dataframe 

with open('our_labels.json') as json_data:
    d = json.load(json_data)
    for activity in d:
        activity['sensor_type'] = 'interval_label'
        activity['device_type'] = 'smartphone'
        print activity['label']
        activity['label_start'] = str(df[df['loggingTime(txt)'].str.contains(activity['label_start_datetime'])].iloc[0]['locationTimestamp_since1970(s)'].astype(float)*1000000000)
        activity['label_end'] = str(df[df['loggingTime(txt)'].str.contains(activity['label_end_date_time'])].iloc[0]['locationTimestamp_since1970(s)'].astype(float)*1000000000)
    with open('our_labels.json', 'w') as outfile:
        json.dump(d, outfile)


# Open the created label json and use it to create a pandas dataframe
# then save the dataframe as a CSV to labels_custom
with open('our_labels.json') as json_data:
    d = json.load(json_data)
    labels_dataframe = pd.read_json('our_labels.json')
    # labels_dataframe = pd.DataFrame.from_dict(labels, orient='index')
    labels_dataframe.to_csv('../our_dataset/labels_custom.csv')