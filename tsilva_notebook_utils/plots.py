
import numpy as np
from typing import Union, List, Optional, Any, Callable

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
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool
    from sklearn.manifold import TSNE, PCA

    if output_notebook:
        from bokeh.io import output_notebook
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
            from sklearn.manifold import TSNE
            embedder = TSNE(n_components=2, random_state=random_state, **embed_kwargs)
        elif embedding_method == "umap":
            import umap
            embedder = umap.UMAP(n_components=2, random_state=random_state, **embed_kwargs)
        elif embedding_method == "pca":
            from sklearn.decomposition import PCA
            embedder = PCA(n_components=2, **embed_kwargs)
        else:
            raise ValueError(f"Unsupported method: {embedding_method}")
        embedding_2d = embedder.fit_transform(data)
    else:
        embedding_2d = embedding_method.fit_transform(data)

    # --- Input Renderer ---
    def default_renderer(inp):
        from PIL.Image import Image
        from io import BytesIO
        import base64
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
