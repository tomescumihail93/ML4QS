import pandas as pd
import numpy as np

df = pd.read_csv('../our_dataset/our_dataset_ns.csv')

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

acc_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[0] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['locationTimestamp_since1970(s)'].astype(float)*1000000000,
        'x': df['accelerometerAccelerationX(G)'],
        'y': df['accelerometerAccelerationY(G)'],
        'z': df['accelerometerAccelerationZ(G)']
    }
)

gyro_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[1] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['locationTimestamp_since1970(s)'].astype(float)*1000000000,
        'x': df['gyroRotationX(rad/s)'],
        'y': df['gyroRotationY(rad/s)'],
        'z': df['gyroRotationZ(rad/s)']
    }
)

mag_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[2] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['locationTimestamp_since1970(s)'].astype(float)*1000000000,
        'x': df['motionMagneticFieldX(uT)'],
        'y': df['motionMagneticFieldY(uT)'],
        'z': df['motionMagneticFieldZ(uT)']
    }
)

press_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[3] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['locationTimestamp_since1970(s)'].astype(float)*1000000000,
        'pressure': df['altimeterPressure(kPa)'],
    }
)

pedom_dataframe = pd.DataFrame(
    {
        'sensor_type': sensors[4] * DATA_FRAME_SIZE,
        'device_type': device_type * DATA_FRAME_SIZE,
        'timestamps': df['locationTimestamp_since1970(s)'].astype(float)*1000000000,
        'steps': df['pedometerNumberofSteps(N)'],
        'distance': df['pedometerDistance(m)']
    }
)

acc_dataframe.to_csv('../our_dataset/acc_custom.csv')
gyro_dataframe.to_csv('../our_dataset/gyro_custom.csv')
mag_dataframe.to_csv('../our_dataset/mag_custom.csv')
press_dataframe.to_csv('../our_dataset/press_custom.csv')
pedom_dataframe.to_csv('../our_dataset/pedom_custom.csv')