import default
import matplotlib.pyplot as plt
import numpy as np
import mplcursors  # Library for cursor hover functionality
import matplotlib

# Use Agg backend when running without a GUI
if not matplotlib.rcParams['backend']: # TODO TODO TODO
    matplotlib.use('Agg')

def plot_FID(signal: np.ndarray, time: np.ndarray, title: str = "No Title") -> None:
    plt.plot(time, np.abs(signal), linestyle='-', color='black', linewidth=0.5)  # marker='.', markersize=0.2,
    plt.xlabel("Time")
    plt.title(title)
    plt.grid(which='both', linestyle=':', linewidth=0.5)
    cursor = mplcursors.cursor(hover=True)  # Enable cursor hover

    # Check if backend is 'agg' to save as SVG
    plt.savefig('plot.svg') if matplotlib.rcParams['backend'] == 'agg' else plt.show()


def plot_FIDs(signal: dict[str, np.ndarray], time: np.ndarray, save_to_file: bool = False) -> None:
    # Create a figure and subplots with reduced vertical space
    num_subplots = len(signal)
    fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 6), sharex=True)
    plt.subplots_adjust(hspace=0)  # Set the vertical space between subplots to zero

    # Iterate through the dictionary keys and arrays
    for i, (key, value) in enumerate(signal.items()):
        axs[i].plot(time, np.abs(value), linestyle='-', color='black', linewidth=0.5)  # marker='.', markersize=0.2,

        # Add title as text over the plot
        axs[i].text(1, 0.5, key, transform=axs[i].transAxes, fontsize='medium', ha='right', va='center', rotation=0)
        axs[i].grid(which='both', linestyle=':', linewidth=0.5)
        axs[i].minorticks_on()

    axs[-1].set_xlabel('Time')  # Set common x-axis label
    plt.tight_layout(h_pad=-0.5)  # Adjust layout for better spacing
    if matplotlib.rcParams['backend'] != 'agg': plt.show()
    cursor = mplcursors.cursor(hover=True)  # Enable cursor hover

    # Add annotation to display data values on hover
    for i, (key, value) in enumerate(signal.items()):
        cursor.connect("add", lambda sel, key=key, value=value: sel.annotation.set_text(f'{key}: {value[sel.target.index]:.4f}'))

    # Display the plot
    plt.savefig('plot.svg') if (matplotlib.rcParams['backend'] == 'agg' or save_to_file is True) else plt.show()

## Plot frontal, sagittal, transverse plane
#    import matplotlib.pyplot as plt
#    import matplotlib

#    # Use Agg backend when running without a GUI
#    if not matplotlib.rcParams['backend']:  # TODO TODO TODO
#        matplotlib.use('Agg')

#    # x, y
#    plt.imshow(np.abs(mrsi_data[:,:,0,0,50]))
#    plt.savefig('plot1.svg')

#    # x, z
#    plt.imshow(np.abs(mrsi_data[:,20,:,0,50]))
#    plt.savefig('plot2.svg')

#    # y, z
#    plt.imshow(np.abs(mrsi_data[20,:,:,0,50]))
#    plt.savefig('plot3.svg')

## Maybe 3D Plot
# display brain planes of 3d image --> https://www.geeksforgeeks.org/three-dimensional-plotting-in-python-using-matplotlib/
# maybe class: Plot::FID -> FID(dictionary with key and values as np.arrays, time_vector)