import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys

from status_bar import print_progress_bar
from datetime import datetime, timedelta
from meteostat import Stations, Hourly, Point
from scipy.optimize import curve_fit

stations = Stations()

location = Point(49.933, -97.0703, 70)
start = datetime(2018, 1, 1)
end = datetime(2021, 12, 30)
start_simulation = start
end_simulation = end
# Get daily weather data for a specific time range
data = Hourly(location, start, end)
data = data.fetch()

difference = end - start
N = int(difference.total_seconds()/60)

window_size = 60  # 60 minutes moving average
interpolated_data = data.resample('min').interpolate(method='linear').rolling(window=window_size).mean()
data = interpolated_data.fillna(data.resample('min').interpolate(method='linear'))

df = pd.DataFrame(index=pd.date_range(start, end, freq='min'), columns=['outside_air_temp', 'return_air_temp', 'mixed_air_damper_position', 'cooling_valve_position', 'heating_valve_position', 'discharge_air_temp', 'econ_state'])

# Discharge setpoint
discharge_setpoint = 16  # degrees C

# Initialize error values for PID controller
integral = 0
previous_error = 0

# PID controller parameters
Kp = 0.02
Ki = 0.01
Kd = 0.00

time = start_simulation
# Initialize cooling and heating valve position and discharge air temperature
df.loc[time, 'mixed_air_damper_position'] = 0
df.loc[time, 'cooling_valve_position'] = 0
df.loc[time, 'heating_valve_position'] = 0
df.loc[time, 'discharge_air_temp'] = 20
econ_state = False
df.loc[time, 'econ_state'] = int(econ_state)
# Economizer setpoint and differential
economizer_setpoint = 15  # degrees C
economizer_differential = 2  # degrees C
min_mad_pos = .2

# Initialize outside air temperature variables
avg_temp = 10  # degrees C
temp_amplitude = 10  # degrees C

# Initialize moving average variables
avg_temp_moving_avg = avg_temp
temp_amplitude_moving_avg = temp_amplitude

# Preallocate arrays for the new values
new_values = np.empty((N+1, len(df.columns)), dtype=np.float64)

# Simplified model of discharge air temperature and cooling valve position:
for i in range(1, N):

    if i % 100 == 0 or i == N:
        print_progress_bar(i, N+1, prefix = f'Simulating {N} minutes from: {start} to: {end}', suffix = 'Complete', length = 40)

    time = start_simulation + timedelta(minutes=i)  # current time in datetime
    prev_time = time - timedelta(minutes=1)  # previous time in datetime

    # Single dataframe lookup at 'time' and 'prev_time'
    oat = data.loc[time, 'temp']
    previous_dat = new_values[i-1, df.columns.get_loc('discharge_air_temp')]

    # Compute values
    rat = 24 + .2*oat + np.random.normal(scale=.5)
    error = previous_dat - discharge_setpoint
    integral += error
    derivative = error - previous_error
    pid_output = Kp*error + Ki*integral + Kd*derivative

    mad_pos, c_pos, h_pos = 0, 0, 0
    if pid_output < 1 and pid_output > 0:
        mad_pos = pid_output
    elif pid_output > 1:
        mad_pos = 1
        c_pos = pid_output - 1
    elif pid_output < 0:
        mad_pos = min_mad_pos
        h_pos = - pid_output

    econ_state = oat > economizer_setpoint - economizer_differential or (econ_state and oat >= economizer_setpoint)
    if econ_state:
        mad_pos = min_mad_pos

    # Clip values
    c_pos = np.clip(c_pos, 0, 1)
    h_pos = np.clip(h_pos, 0, 1)
    mad_pos = np.clip(mad_pos, min_mad_pos, 1)
    econ_state = int(econ_state)

    # Update discharge air temperature based on new cooling and heating valve position
    dat = (
        rat*(0.7 - 0.4*mad_pos) 
        + oat*(0.3 + 0.4*mad_pos) 
        - 20*c_pos
        + 20*h_pos
    )

    # Store values in new_values array
    new_values[i, :] = [oat, rat, mad_pos, c_pos, h_pos, dat, econ_state]
    previous_error = error

# Single save to dataframe
df.loc[start_simulation + timedelta(minutes=1):end] = new_values[1:]
df.to_pickle('prepared_data.pkl')    

fig, axs = plt.subplots(2, 1, figsize=(12, 8))  # Create two subplots: 2 rows, 1 column

# Plot other variables in the second subplot
df[['outside_air_temp', 'return_air_temp', 'discharge_air_temp']].plot(ax=axs[0])
axs[0].set_title('Temperatures')
axs[0].set_ylabel('Degrees C')

# Plot variables with range 0-1 in the first subplot
df[['mixed_air_damper_position', 'cooling_valve_position', 'heating_valve_position', 'econ_state']].plot(ax=axs[1])
axs[1].set_title('Valve and Damper Positions')
axs[1].set_ylabel('Value')

plt.tight_layout()
plt.show()
