import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of time steps
N = 24*60  # minutes in a day
N = N*2 # 7 days

# Time variable (in hours)
t = np.arange(N) / 60

# Initialize dataframe
df = pd.DataFrame(index=t, columns=['outside_air_temp', 'return_air_temp', 'mixed_air_damper_position', 'cooling_valve_position', 'heating_valve_position', 'discharge_air_temp'])

# Outside air temperature (cyclical over 24 hours, with some random variation)
avg_temp = 24  # degrees C
temp_amplitude = 10  # degrees C
df['outside_air_temp'] = avg_temp + temp_amplitude * np.sin(2*np.pi*(t/24)) + np.random.normal(scale=.5, size=N)

# Return air temperature (somewhat arbitrary function of outside air temp)
df['return_air_temp'] = 24 + .1*df['outside_air_temp'] + np.random.normal(scale=.5, size=N)

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
df.loc[0, 'discharge_air_temp'] = df.loc[0, 'return_air_temp']
econ_state = False
# Economizer setpoint and differential
economizer_setpoint = 20  # degrees C
economizer_differential = 2  # degrees C

# Simplified model of discharge air temperature and cooling valve position:
for i in range(1, N):
    time = i / 60  # current time in hours
    prev_time = (i - 1) / 60  # previous time in hours

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
        df.loc[time, 'mixed_air_damper_position'] = 0
        df.loc[time, 'cooling_valve_position'] = 0
        df.loc[time, 'heating_valve_position'] = - pid_output
    
    # Check outside air temperature against economizer setpoint and differential
    if df.loc[time, 'outside_air_temp'] < economizer_setpoint - economizer_differential:
        econ_state = True
    elif df.loc[time, 'outside_air_temp'] > economizer_setpoint:
        econ_state = False

    # Update mixed air damper position based on economizer state
    if econ_state:
        df.loc[time, 'mixed_air_damper_position'] = 0

    # Clip the cooling valve position between 0 and 1
    df.loc[time, 'cooling_valve_position'] = np.clip(df.loc[time, 'cooling_valve_position'], 0, 1)
    # Clip the heating valve position between 0 and 1
    df.loc[time, 'heating_valve_position'] = np.clip(df.loc[time, 'heating_valve_position'], 0, 1)
    # Clip the heating valve position between 0 and 1
    df.loc[time, 'mixed_air_damper_position'] = np.clip(df.loc[time, 'mixed_air_damper_position'], 0, 1)

    previous_error = error
    
    # Update discharge air temperature based on new cooling and heating valve position
    df.loc[time, 'discharge_air_temp'] = (
        df.loc[time, 'return_air_temp']*0.7 
        + df.loc[time, 'outside_air_temp']*0.3*df.loc[time, 'mixed_air_damper_position'] 
        - 20*df.loc[time, 'cooling_valve_position']
        + 20*df.loc[time, 'heating_valve_position']
    )

# Plotting the dataframe
df.plot()
plt.show()
