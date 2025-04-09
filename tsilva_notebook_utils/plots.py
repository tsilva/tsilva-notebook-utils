def plot_line(
    data,
    title='Series Plot',
    xlabel='X-axis', 
    ylabel='Y-axis',
    figsize=(12, 6)
):
    import matplotlib.pyplot as plt

    # Create a new figure for plotting
    plt.figure(figsize=figsize)

    # Plot the data
    plt.plot(data)
    
    # Add title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    return plt

def plot_series(
    data, 
    labels=None, 
    title='Series Plot',
    xlabel='X-axis', 
    ylabel='Y-axis',
    figsize=(12, 6)
):
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure data is a NumPy array
    data = np.array(data)

    # Create a new figure for plotting
    plt.figure(figsize=figsize)

    n_series = data.shape[1]

    # Plot each series
    for i in range(n_series):
        label = labels[i] if labels and i < len(labels) else f'Series {i}'
        plt.plot(data[:, i], label=label)

    # Add title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add legend outside the plot area if labels are provided
    if labels:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')

    # Adjust layout to make room for the legend and labels
    plt.tight_layout()

    return plt

def plot_histogram(
    data, 
    title='Histogram', 
    bins=50
):
    import matplotlib.pyplot as plt

    plt.hist(data, bins=bins)
    plt.title(title)

    return plt

def plot_training_loss(data):
    return plot_line(
        data, 
        title="Training Loss", 
        xlabel="Epoch", 
        ylabel="Loss"
    )

def plot_series_over_indices(
    data_map, 
    title='Series over Indices',
    xlabel='Index',
    ylabel='Value'
):
    import numpy as np

    # Stack the data from the map into a 2D array (shape: indices x series)
    data_array = np.vstack(list(data_map.values())).T
    labels = list(data_map.keys())

    return plot_series(
        data_array,
        labels=labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )

def plot_snapshots_over_indices(
    data, 
    key, 
    title='Snapshots over Indices',
    xlabel='Index',
    ylabel='Value',
    serieslabel='Series',
    num_snapshots=5
):
    import numpy as np

    total_indices = len(data[key])
    snapshot_indices = np.linspace(0, total_indices - 1, num=num_snapshots, dtype=int)

    # Collect data at snapshot indices
    snapshots = [data[key][i] for i in snapshot_indices]
    data_array = np.array(snapshots).T
    labels = [f'{serieslabel} {i}' for i in snapshot_indices]

    return plot_series(
        data_array,
        labels=labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
