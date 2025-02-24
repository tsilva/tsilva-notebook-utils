# tsilva-notebook-utils

A collection of utility functions for Jupyter/Colab notebooks.

## Installation

```bash
pip install tsilva-notebook-utils
```

## Usage

### Video Rendering

The package provides two main functions for rendering videos in notebooks:

#### Single Video Rendering

```python
from tsilva_notebook_utils import render_video

# Render a sequence of frames
frames = [frame1, frame2, frame3]  # numpy arrays (images)
render_video(frames, fps=30, scale=1.0)

# Render frames with labels
labeled_frames = [(frame1, "Frame 1"), (frame2, "Frame 2")]
render_video(labeled_frames, fps=30, scale=1.0)
```

Parameters:
- `frames`: List of frames (numpy arrays) or tuples of (frame, label)
- `scale`: Scale factor for video dimensions (default: 1.0)
- `fps`: Frames per second (default: 30)
- `format`: Video format (default: 'mp4')
- `font_scale`: Font scale for text labels (default: 0.5)

#### Multiple Videos Side by Side

```python
from tsilva_notebook_utils import render_videos

# Render multiple videos with labels
video_tuples = [
    (frames1, "Video 1"),
    (frames2, "Video 2")
]
render_videos(video_tuples, fps=30, scale=1.0)
```
