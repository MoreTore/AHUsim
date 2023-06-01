import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
from enum import Enum
import random
import time
from matplotlib.lines import Line2D
from status_bar import print_progress_bar
from datetime import datetime, timedelta
from meteostat import Stations, Hourly, Point
from scipy.optimize import curve_fit

from keras.models import load_model
import tensorflow_addons as tfa
import joblib

model = load_model('models/autoencoder_modelV2_retrained.h5', custom_objects={'AdamW': tfa.optimizers.AdamW})
scaler = joblib.load('models/scalerV2.2.pkl')


class FaultTypes(Enum):
    NONE = 0
    OUTSIDE_AIR_TEMP = 1
    RETURN_AIR_TEMP = 2
    MIXED_AIR_DAMPER_POSITION = 3
    COOLING_VALVE_POSITION = 4
    HEATING_VALVE_POSITION = 5

fault_types = list(FaultTypes)
fault_timer = 0
fault_type = FaultTypes.NONE

# AHU setpoints
economizer_setpoint = 15  # degrees C
economizer_differential = 2  # degrees C
min_mad_pos = .2

# Simulation settings
start = datetime(2018, 1, 1)
end = datetime(2019, 12, 28)
t_now = start
N = int((end - start).total_seconds()/60)
# Fault generation parameters
fault_prob = 0.01  # Probability of a fault occurring at any given minute
fault_duration = 3600  # Duration of a fault in minutes

# Get hourly weather data for a specific time range
#stations = Stations()
location = Point(49.933, -97.0703, 70)
data = Hourly(location, start, end)
data = data.fetch()
interpolated_data = data.resample('min').interpolate(method='linear').rolling(window=60).mean()
data = interpolated_data.fillna(data.resample('min').interpolate(method='linear'))

# Preallocate arrays for the new values
columns = ['outside_air_temp',
           'return_air_temp', 
           'mixed_air_damper_position', 
           'cooling_valve_position', 
           'heating_valve_position', 
           'discharge_air_temp', 
           'econ_state', 
           'fault_type']
RANDOM_SIZE_INTERATIONS = 10
RANDOM_DAT_SP_RANDOM_SIZE_INTERATIONS = 10
TOT_ITER = 100*len(fault_types)*N

# Define constants
AIR_DENSITY = 1.225 # kg/m^3 at sea level and 15 degrees Celsius
SPECIFIC_HEAT_CAPACITY_AIR = 1005 # J/(kg*K) for air at constant pressure
AIR_FLOW_RATE = 20 # m^3/s, define this as per your system's specs
MASS_FLOW_RATE_AIR = AIR_DENSITY * AIR_FLOW_RATE # kg/s

start_time = time.time()
completed_iterations = 0
print(f'Total simulation minutes: {TOT_ITER}')

def do_progress_bar():
    elapsed_time = time.time() - start_time
    avg_time_per_iteration = elapsed_time / completed_iterations
    remaining_iterations = TOT_ITER - completed_iterations
    eta = remaining_iterations * avg_time_per_iteration
    eta = timedelta(seconds=eta)

    hours = eta.seconds // 3600
    minutes = (eta.seconds // 60) % 60
    seconds = eta.seconds % 60

    # This will give you a time format like "2:30:45" for 2 hours, 30 minutes, and 45 seconds.
    eta_time = "%d:%02d:%02d" % (hours, minutes, seconds)
    print_progress_bar(i, N+1, prefix = f'\rSimulating {N} minutes from: {start} to: {end}', suffix = f'Complete. ETA: {eta_time}, {fault_type}', length = 10)

def reset():
    pass
    

for n in range(100):
    discharge_setpoint = round(random.uniform(10, 24), 1)  # degrees C
    cooling_valve_size = round(random.uniform(1, 2), 1)
    heating_valve_size = round(random.uniform(1, 2), 1)
    hold_fault_state = False

    # PID controller parameters
    Kp = 0.01
    Ki = 0.005
    Kd = 0.00
    integral = 0
    previous_error = 0
    error = 0
    df = pd.DataFrame(index=pd.date_range(start, end, freq='min'), columns=columns, dtype=np.float16)
    new_values = np.zeros((N+1, len(columns)), dtype=np.float16)
    # Initialize
    initial_conditions = [0, 0, 0, 0, 20, 0]
    df.loc[t_now, columns[1:7]] = initial_conditions
    new_values[0, 1:7] = initial_conditions
    # Ensure 'data' has the same length as 'new_values'
    if len(data) == len(new_values):
        # Assign 'outside_air_temp' from 'data' to 'new_values'
        new_values[:, 0] = data['temp'].values
    else:
        print("Data sizes do not match")
    econ_state = 0
    previous_dat = 0

    for i in range(1, N):
        completed_iterations += 1
        
        if i % 5000 == 0 or i == N:
            do_progress_bar()
        
        if np.random.random() < fault_prob and fault_timer == 0:
            # Start a new fault
            fault_timer = fault_duration
            fault_type = np.random.choice(fault_types)
            hold_fault_state = False

        t_now = start + timedelta(minutes=i)  # current time in datetime
        prev_time = t_now - timedelta(minutes=1)  # previous time in datetime

        # Single dataframe lookup at 'time' and 'prev_time'
        oat = new_values[i][0]

        # Compute values
        rat = 24 + .2*oat + np.random.normal(scale=.5)
        error = previous_dat - discharge_setpoint
        integral += error
        if integral > 1000:
            integral = 1000

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

        f_oat = r_oat = oat
        f_h_pos = r_h_pos = h_pos
        f_c_pos = r_c_pos = c_pos
        f_mad_pos = r_mad_pos = mad_pos
        f_rat = r_rat = rat

        if fault_timer > 0:
            # We're currently simulating a fault
            if fault_type == FaultTypes.OUTSIDE_AIR_TEMP:
                if not hold_fault_state:
                    hold_value = round(random.uniform(-40, 40), 3)
                    hold_fault_state = True
                f_oat = hold_value
            elif fault_type == FaultTypes.RETURN_AIR_TEMP:
                if not hold_fault_state:
                    hold_value = round(random.uniform(-40, 40), 3)
                    hold_fault_state = True
                f_rat = hold_value
            elif fault_type == FaultTypes.MIXED_AIR_DAMPER_POSITION:
                if not hold_fault_state:
                    hold_value = random.random()
                    hold_fault_state = True
                f_mad_pos = hold_value
            elif fault_type == FaultTypes.COOLING_VALVE_POSITION:
                if not hold_fault_state:
                    hold_value = random.choice([0, 1])
                    hold_fault_state = True
                f_c_pos = hold_value
            elif fault_type == FaultTypes.HEATING_VALVE_POSITION:
                if not hold_fault_state:
                    hold_value = random.choice([0, 1])
                    hold_fault_state = True
                f_h_pos = hold_value
            elif fault_type == FaultTypes.NONE:
                pass
            fault_timer -= 1

        # Add noise to the heating and cooling loop temperatures
        heating_loop_temp = 35 + round(random.uniform(-2, math.sin(random.random())), 3) + math.sin(random.random())
        cooling_water_temp = 8 + round(random.uniform(-2, math.sin(random.random())), 3) + math.sin(random.random())

        # Calculate mixed air temperature based on RAT, OAT, and damper position
        mixed_air_temp = r_rat * (1 - f_mad_pos) + r_oat * f_mad_pos

        # Calculate the maximum possible cooling and heating
        max_possible_cooling = max(mixed_air_temp - max(oat, cooling_water_temp), 0)
        max_possible_heating = max(heating_loop_temp - mixed_air_temp, 0)

        # Calculate potential heat transfer for cooling and heating
        potential_cooling = cooling_valve_size * f_c_pos * MASS_FLOW_RATE_AIR * SPECIFIC_HEAT_CAPACITY_AIR * max_possible_cooling
        potential_heating = heating_valve_size * f_h_pos * MASS_FLOW_RATE_AIR * SPECIFIC_HEAT_CAPACITY_AIR * max_possible_heating

        # Calculate the mixed air temperature component of potential DAT
        mixed_air_temp_component = mixed_air_temp * MASS_FLOW_RATE_AIR * SPECIFIC_HEAT_CAPACITY_AIR
        
        # Calculate potential DAT considering heat transfer equation
        dat = (mixed_air_temp_component - potential_cooling + potential_heating) / (MASS_FLOW_RATE_AIR * SPECIFIC_HEAT_CAPACITY_AIR)

        # Enforce constraints of not exceeding heating loop temperature and not dropping below the maximum of oat and cooling water temperature
        lower_limit = max(oat, cooling_water_temp)
        upper_limit = heating_loop_temp

        if dat < lower_limit:
            dat = lower_limit
        elif dat > upper_limit:
            dat = upper_limit        

        # Store values in new_values array
        new_values[i, :] = [round(x, 3) for x in [f_oat, f_rat, r_mad_pos, r_c_pos, r_h_pos, dat]] + [econ_state, fault_type.value]
        previous_dat = dat
        previous_error = error
        SEQUENCE_LENGTH = 60  # This value can be tuned during training
        if i > SEQUENCE_LENGTH+1:
            # Remove the last column
            eval_values = new_values[(i-60):i, :-1]
            df_slice_normalized = scaler.fit_transform(eval_values)
            current_slice = np.array([df_slice_normalized])
            prediction = model.predict(current_slice)
            print(prediction)
            predicted_fault_type = np.argmax(prediction, axis=1)
            print(FaultTypes(predicted_fault_type[0]))
            print(fault_type)
            if FaultTypes(predicted_fault_type[0]) == fault_type:
                print("true")

    # Single save to dataframe
    df.loc[start + timedelta(minutes=1):end] = new_values[1:]
    df.to_pickle(f'data/prepared_data_{n}_{discharge_setpoint}_{cooling_valve_size}_{heating_valve_size}.pkl')

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))  # Create three subplots: 3 rows, 1 column

    # Plot other variables in the second subplot
    df[['outside_air_temp', 'return_air_temp', 'discharge_air_temp']].plot(ax=axs[0])
    axs[0].set_title('Temperatures')
    axs[0].set_ylabel('Degrees C')

    # Plot variables with range 0-1 in the first subplot
    df[['mixed_air_damper_position', 'cooling_valve_position', 'heating_valve_position', 'econ_state']].plot(ax=axs[1])
    axs[1].set_title('Valve and Damper Positions')
    axs[1].set_ylabel('Value')

    df[['fault_type']].plot(ax=axs[2])
    axs[2].set_title('Fault Type')
    axs[2].set_ylabel('Value')
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'{fault_type.value}: {fault_type.name}') for fault_type in FaultTypes]

    # Plot the legend
    plt.legend(handles=legend_elements, title='Fault Type', loc='lower right')



    fig.suptitle(f'Simulation number {n} DA-T:{discharge_setpoint} CLG_VLV SIZE:{cooling_valve_size} HTG_VLV SIZE:{heating_valve_size}')

    plt.tight_layout()
    plt.savefig('plots/current.png')
    plt.savefig(f'plots/simulate_errors_v2_{n}_{discharge_setpoint}_{cooling_valve_size}_{heating_valve_size}.png')
    plt.close()