import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

# We'll need to plot an empty line first and update it later in the animate function
line, = ax.plot([], [])

# Initialization function for the plot. 
# Here we set the x and y data to empty lists, since we'll be appending data as we go.
def init():
    ax.set_xlim(start_simulation, start_simulation + timedelta(seconds=N))  # set x-axis limits
    ax.set_ylim(0, 40)  # set y-axis limits
    line.set_data([], [])
    return line,

# The function to animate the plot. It gets called in each iteration of FuncAnimation
def animate(i):
    time = start_simulation + timedelta(seconds=i)  # current time in datetime
    if time in data.index:
        df.loc[time, 'outside_air_temp'] = oat = data.loc[time, 'temp']
    else:
        df.loc[time, 'outside_air_temp'] = np.nan  # or some other value
    
    # Here we add new data to the plot
    xdata = df.index
    ydata = df['outside_air_temp']
    line.set_data(xdata, ydata)
    return line,

# This is the FuncAnimation call. We pass in the figure, animate function, init function, and the frames argument is how many times animate will be called.
ani = FuncAnimation(fig, animate, frames=N, init_func=init, blit=True)

# Finally, we display the plot
plt.show()