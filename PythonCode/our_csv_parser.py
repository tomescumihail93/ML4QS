# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv('../our_dataset/our_data_last.csv', parse_dates=['locationTimestamp_since1970(s)'])

TOTAL_SENSORS = 5
DATA_FRAME_SIZE = len(df)

device_type = ['smartphone']
sensors = [
    ['accelerometer'],
    ['gyroscope'],
    ['magnetometer'],
    ['pressure'],
    ['pedometer']
]
# index = pd.DatetimeIndex('loggingTime(txt)')
# print(index)
# pd.to_numeric('locationTimestamp_since1970(s)', errors='coerce')
# pd.to_numeric('locationTimestamp_since1970(s)', errors='coerce').isnull()

acc_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[0] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['loggingTime(txt)']
,
        'x': df['accelerometerAccelerationX(G)']*10,
        'y': df['accelerometerAccelerationY(G)']*10,
        'z': df['accelerometerAccelerationZ(G)']*10
    }
)

gyro_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[1] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['loggingTime(txt)'],
        'x': df['gyroRotationX(rad/s)'],
        'y': df['gyroRotationY(rad/s)'],
        'z': df['gyroRotationZ(rad/s)']
    }
)

mag_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[2] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['loggingTime(txt)'],
        'x': df['motionMagneticFieldX(µT)'],
        'y': df['motionMagneticFieldY(µT)'],
        'z': df['motionMagneticFieldZ(µT)']
    }
)

press_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[3] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['loggingTime(txt)'],
        'pressure': df['altimeterPressure(kPa)'],
    }
)

pedom_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[4] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['loggingTime(txt)'],
        'steps': df['pedometerNumberofSteps(N)'],
        'distance': df['pedometerDistance(m)']
    }
)

acc_dataframe.to_csv('../our_dataset/acc_custom.csv')
gyro_dataframe.to_csv('../our_dataset/gyro_custom.csv')
mag_dataframe.to_csv('../our_dataset/mag_custom.csv')
press_dataframe.to_csv('../our_dataset/press_custom.csv')
pedom_dataframe.to_csv('../our_dataset/pedom_custom.csv')