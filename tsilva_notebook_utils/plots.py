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

def plot_gradients_over_epochs(
    gradients_dict, 
    title='Gradient Norms over Epochs'
):
    import numpy as np

    # Stack the gradients
    gradients_array = np.vstack(list(gradients_dict.values())).T
    labels = list(gradients_dict.keys())

    # Plot
    return plot_series(
        gradients_array,
        labels=labels,
        title=title,
        xlabel='Epoch',
        ylabel='Gradient Norm'
    )

def plot_gradients_over_time_steps(
    gradients_over_time, 
    key, 
    title='Gradients over Time Steps'
):
    import numpy as np
    from tsilva_notebook_utils import plot_series

    # Determine total epochs
    total_epochs = len(gradients_over_time[key])

    # Define representative epochs to plot
    epochs_to_plot = [
        0,
        max(1, total_epochs // 4),
        max(1, total_epochs // 2),
        max(1, (3 * total_epochs) // 4),
        total_epochs - 1
    ]

    # Collect gradient norms for selected epochs
    gradients_list = [gradients_over_time[key][epoch] for epoch in epochs_to_plot]

    # Convert to NumPy array with shape (time_steps, n_series)
    gradients_array = np.array(gradients_list).T

    # Create labels for the series
    labels = [f'Epoch {epoch}' for epoch in epochs_to_plot]

    # Plot using the reusable function
    return plot_series(
        gradients_array,
        labels=labels,
        title=title,
        xlabel='Time step',
        ylabel='Gradient Norm'
    )
