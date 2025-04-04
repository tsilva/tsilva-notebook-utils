# 🧰 tsilva-notebook-utils

🔬 Handy utilities for enhancing your Jupyter and Google Colab notebooks

## 📖 Overview

`tsilva-notebook-utils` is a collection of utility functions designed to make working with Jupyter and Google Colab notebooks more efficient. It provides tools for video rendering, notification systems, and Colab-specific features like automatic disconnection after idle periods.

## 🛠️ Usage

### Video Rendering

```python
from tsilva_notebook_utils import render_video

# Render a simple video from frames
frames = [frame1, frame2, frame3]  # List of numpy arrays
video = render_video(frames, fps=30, scale=1.5)
display(video)

# Render frames with labels
labeled_frames = [(frame1, "Start"), (frame2, "Middle"), (frame3, "End")]
video = render_video(labeled_frames, fps=24)
display(video)

# Compare multiple videos side by side
from tsilva_notebook_utils import render_videos
render_videos([(video1_frames, "Original"), (video2_frames, "Processed")])
```

### Google Colab Utilities

```python
from tsilva_notebook_utils import disconnect_after_timeout

# Automatically disconnect Colab after 5 minutes of inactivity
disconnect_after_timeout(timeout_seconds=300)
```

### Notifications

Send notifications to [PopDesk](https://github.com/tsilva/popdesk) notification server:

```python
from tsilva_notebook_utils import send_popdesk_notification

# Send a notification when your long-running notebook task completes
send_popdesk_notification(
    url="https://your-popdesk-url",
    auth_token="your-auth-token",
    title="Training Complete",
    message="Your model has finished training with 95% accuracy"
)
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).