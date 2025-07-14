
import base64
from io import BytesIO
from typing import Any, Callable, List, Optional, Union
import numpy as np

from .numpy import reshape_vector_to_grid, to_numpy


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
    import matplotlib.pyplot as plt
    import numpy as np
    

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

def plot_loss_curve(data):
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

def plot_embeddings_with_inputs(
    data: Union[np.ndarray, List[Any]],
    raw_inputs: List[Any],
    input_renderer: Optional[Callable[[Any], str]] = None,
    embedding_method: Union[str, Any] = "tsne",
    point_size: int = 5,
    plot_title: str = "2D Embedding Visualization",
    captions: Optional[List[str]] = None,
    random_state: int = 42,
    output_notebook=True,
    **embed_kwargs
):
    """
    Visualizes embeddings in 2D with hoverable input representations (images, text, tensors, etc.)

    Args:
        data: Feature vectors (or embeddings) to reduce to 2D.
        raw_inputs: Original inputs (images, tensors, strings, etc.).
        input_renderer: Function to convert an input into an HTML string (e.g., image, preview text).
        embedding_method: Embedding reducer - 'tsne', 'umap', 'pca', or custom.
        point_size: Size of scatter points.
        plot_title: Plot title.
        captions: Optional text labels.
        random_state: Random seed.
        **embed_kwargs: Passed to the embedding transformer.
    """
    

    from PIL.Image import Image
    import numpy as np
    import umap
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from bokeh.io import output_notebook
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure, show

    if output_notebook:
        output_notebook()

    if raw_inputs is None or len(raw_inputs) == 0:
        raise ValueError("raw_inputs must be non-empty.")
    if data is None or (isinstance(data, np.ndarray) and data.size == 0):
        raise ValueError("data must be a non-empty array or list.")
    if captions and len(captions) != len(raw_inputs):
        raise ValueError("captions must match raw_inputs in length.")

    # --- Compute Embedding ---
    if isinstance(embedding_method, str):
        embedding_method = embedding_method.lower()
        if embedding_method == "tsne":
            embedder = TSNE(n_components=2, random_state=random_state, **embed_kwargs)
        elif embedding_method == "umap":
            embedder = umap.UMAP(n_components=2, random_state=random_state, **embed_kwargs)
        elif embedding_method == "pca":
            embedder = PCA(n_components=2, **embed_kwargs)
        else:
            raise ValueError(f"Unsupported method: {embedding_method}")
        embedding_2d = embedder.fit_transform(data)
    else:
        embedding_2d = embedding_method.fit_transform(data)

    # --- Input Renderer ---
    def default_renderer(inp):
        
        if isinstance(inp, Image):
            buf = BytesIO(); inp.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"<img src='data:image/png;base64,{b64}' width='64' height='64'>"
        elif isinstance(inp, (str, int, float)):
            return f"<pre>{str(inp)}</pre>"
        elif isinstance(inp, np.ndarray):
            return f"<pre>{np.array2string(inp, precision=2, threshold=5)}</pre>"
        else:
            return f"<pre>{str(inp)}</pre>"

    render_func = input_renderer or default_renderer

    # --- Create Data Source ---
    source = ColumnDataSource(data=dict(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        preview=[render_func(inp) for inp in raw_inputs],
        label=captions or ["" for _ in raw_inputs],
    ))

    hover = HoverTool(tooltips="""
        <div>
            <div><strong>Coords:</strong> ($x{0.2f}, $y{0.2f})</div>
            <div>@preview{safe}</div>
            <div><strong>@label</strong></div>
        </div>
    """, formatters={'$x': 'printf', '$y': 'printf'})

    p = figure(title=plot_title, tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
               width=800, height=600, x_axis_label="Dim 1", y_axis_label="Dim 2")
    p.scatter('x', 'y', source=source, size=point_size, fill_color="navy", alpha=0.6)

    if output_notebook: show(p)
    else: return p


def plot_vector_batch_heatmap(
    tensor_batch,
    title="Vector Batch Heatmap",
    xlabel="Dimensions",
    ylabel="Batch Index",
    cmap="viridis",
    tick_step=10,
    base_width=12,
    base_height_per_row=0.3,
    max_height=12,
    min_height=2
):
    """
    Returns the matplotlib Figure object of the heatmap.
    """
    

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = to_numpy(tensor_batch)

    batch_size, vector_dim = data.shape
    fig_height = np.clip(batch_size * base_height_per_row, min_height, max_height)

    fig, ax = plt.subplots(figsize=(base_width, fig_height))
    sns.heatmap(data, cmap=cmap, cbar=True, ax=ax)

    # X-ticks
    x_step = tick_step if vector_dim > 20 else 1
    x_ticks = list(np.arange(0, vector_dim, x_step))
    if (vector_dim - 1) not in x_ticks:
        x_ticks.append(vector_dim - 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i) for i in x_ticks], rotation=90)

    # Y-ticks
    y_step = tick_step if batch_size > 20 else 1
    y_ticks = list(np.arange(0, batch_size, y_step))
    if (batch_size - 1) not in y_ticks:
        y_ticks.append(batch_size - 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i) for i in y_ticks], rotation=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    return fig

def plot_tensor_stats_heatmaps(
    tensor_batch,
    title_prefix="Latent Vectors",
    collapse_grid_width=16,
    stats=["raw", "mean", "median", "std", "var"],
    **heatmap_kwargs
):

    import torch
    import matplotlib.pyplot as plt
    

    
    assert isinstance(tensor_batch, torch.Tensor), "Input must be a PyTorch tensor"

    stat_funcs = {
        "raw": lambda x: x,
        "mean": lambda x: x.mean(dim=0),
        "median": lambda x: x.median(dim=0).values,
        "std": lambda x: x.std(dim=0, unbiased=False),
        "var": lambda x: x.var(dim=0, unbiased=False),
    }

    figures = {}
    for stat in stats:
        data = stat_funcs[stat](tensor_batch)
        if stat != "raw":
            data = reshape_vector_to_grid(data, max_width=collapse_grid_width)
        fig = plot_vector_batch_heatmap(
            data,
            title=f"{title_prefix} - {stat.capitalize()}",
            **heatmap_kwargs
        )
        figures[stat] = fig
        plt.close(fig)

    return figures

