import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from datetime import datetime
from meteostat import Stations, Daily, Hourly, Point

stations = Stations()

location = Point(49.933, -97.0703, 70)
start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31)
# Get daily weather data for a specific time range
data = Hourly(location, start, end)
data = data.fetch()
# Plot line chart including average, minimum and maximum temperature
data.plot(y=['temp'])
plt.show()
# Number of time steps
N = 24*60  # minutes in a day
N = N*14 # days

# Time variable (in hours)
t = np.arange(N) / 60

# Initialize dataframe
df = pd.DataFrame(index=t, columns=['outside_air_temp', 'return_air_temp', 'mixed_air_damper_position', 'cooling_valve_position', 'heating_valve_position', 'discharge_air_temp', 'econ_state'])

# Discharge setpoint
discharge_setpoint = 16  # degrees C

# Initialize error values for PID controller
integral = 0
previous_error = 0

# PID controller parameters
Kp = 0.02
Ki = 0.01
Kd = 0.00

# Initialize cooling and heating valve position and discharge air temperature
df.loc[0, 'mixed_air_damper_position'] = 0
df.loc[0, 'cooling_valve_position'] = 0
df.loc[0, 'heating_valve_position'] = 0
df.loc[0, 'discharge_air_temp'] = 20
econ_state = False
df.loc[0, 'econ_state'] = int(econ_state)
# Economizer setpoint and differential
economizer_setpoint = 20  # degrees C
economizer_differential = 2  # degrees C
min_mad_pos = .2

# Initialize outside air temperature variables
avg_temp = 10  # degrees C
temp_amplitude = 10  # degrees C

# Initialize moving average variables
avg_temp_moving_avg = avg_temp
temp_amplitude_moving_avg = temp_amplitude


# Simplified model of discharge air temperature and cooling valve position:
for i in range(1, N):
    time = i / 60  # current time in hours
    prev_time = (i - 1) / 60  # previous time in hours

    # Update moving averages with some random variation
    avg_temp_moving_avg += np.random.normal(scale=.2)
    temp_amplitude_moving_avg += np.random.normal(scale=.1)

    # Update outside air temperature with moving averages and cyclical variation
    df.loc[time, 'outside_air_temp'] = avg_temp_moving_avg + temp_amplitude_moving_avg * math.sin(2*np.pi*(time/24)) + np.random.normal(scale=.5)
    
    # Return air temperature (somewhat arbitrary function of outside air temp)
    df.loc[time, 'return_air_temp'] = 24 + .2*df.loc[time, 'outside_air_temp'] + np.random.normal(scale=.5)

    # Calculate error for PID controller
    error =  df.loc[prev_time, 'discharge_air_temp'] - discharge_setpoint
    integral += error
    derivative = error - previous_error
    
    

    # PID output
    pid_output = Kp*error + Ki*integral + Kd*derivative
    
    # Update mixed air damper position and cooling valve position based on PID output
    if pid_output < 1 and pid_output > 0:
        df.loc[time, 'mixed_air_damper_position'] = pid_output
        df.loc[time, 'cooling_valve_position'] = 0
        df.loc[time, 'heating_valve_position'] = 0

    elif pid_output > 1:
        df.loc[time, 'mixed_air_damper_position'] = 1
        df.loc[time, 'cooling_valve_position'] = pid_output - 1
        df.loc[time, 'heating_valve_position'] = 0

    elif pid_output < 0:
        df.loc[time, 'mixed_air_damper_position'] = min_mad_pos
        df.loc[time, 'cooling_valve_position'] = 0
        df.loc[time, 'heating_valve_position'] = - pid_output
    
    # Check outside air temperature against economizer setpoint and differential
    if df.loc[time, 'outside_air_temp'] < economizer_setpoint - economizer_differential:
        econ_state = True
    elif df.loc[time, 'outside_air_temp'] > economizer_setpoint:
        econ_state = False

    # Update mixed air damper position based on economizer state
    if econ_state:
        df.loc[time, 'mixed_air_damper_position'] = min_mad_pos

    df.loc[time, 'econ_state'] = int(econ_state)
    # Clip the cooling valve position between 0 and 1
    df.loc[time, 'cooling_valve_position'] = np.clip(df.loc[time, 'cooling_valve_position'], 0, 1)
    # Clip the heating valve position between 0 and 1
    df.loc[time, 'heating_valve_position'] = np.clip(df.loc[time, 'heating_valve_position'], 0, 1)
    # Clip the heating valve position between 0 and 1
    df.loc[time, 'mixed_air_damper_position'] = np.clip(df.loc[time, 'mixed_air_damper_position'], min_mad_pos, 1)

    previous_error = error
    
    # Update discharge air temperature based on new cooling and heating valve position
    df.loc[time, 'discharge_air_temp'] = (
        df.loc[time, 'return_air_temp']*(0.7 - 0.4*df.loc[time, 'mixed_air_damper_position']) 
        + df.loc[time, 'outside_air_temp']*(0.3 + 0.4*df.loc[time, 'mixed_air_damper_position']) 
        - 20*df.loc[time, 'cooling_valve_position']
        + 20*df.loc[time, 'heating_valve_position']
    )

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
