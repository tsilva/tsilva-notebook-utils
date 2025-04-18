def render_video(
    frames,              # List of frames to render. Each frame can be an image or a tuple (image, label)
    scale=1,             # Scale factor for the video dimensions
    fps=30,              # Frames per second for the video
    format='mp4',        # Video format
    font_scale=0.5       # Font scale for text shown in video corner
):
    import cv2
    import base64
    import imageio
    from io import BytesIO
    from IPython.display import HTML

    def adjust_frame_size(frame, block_size=16):
        """Resize frame dimensions to be divisible by block_size."""
        height, width = frame.shape[:2]
        new_height = (height + block_size - 1) // block_size * block_size
        new_width = (width + block_size - 1) // block_size * block_size
        return cv2.resize(frame, (new_width, new_height))

    # Define text properties once
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)  # Black
    line_type = 1
    margin = 10

    processed_frames = []
    for i, frame in enumerate(frames):
        # Determine if the frame includes a label
        if isinstance(frame, tuple) and len(frame) >= 2:
            image, label = frame
        else:
            image, label = frame, None

        # Adjust frame size
        image = adjust_frame_size(image)
        frame_with_text = image.copy()

        # If there's a label, render it
        if label:
            position = (margin, image.shape[0] - margin)
            font_color = (255, 255, 255) if len(frame_with_text.shape) == 2 else (0, 0, 0)
            cv2.putText(frame_with_text, label, position, font, font_scale, font_color, line_type)

        # Convert BGR to RGB for imageio
        _frame = frame_with_text
        if len(_frame.shape) == 2: _frame = cv2.cvtColor(frame_with_text, cv2.COLOR_GRAY2RGB)
        processed_frames.append(_frame)

    # Write video to buffer by appending frames individually
    buffer = BytesIO()
    with imageio.get_writer(buffer, format=format, fps=fps) as writer:
        for frame in processed_frames:
            writer.append_data(frame)

    # Encode video in base64
    video_data = base64.b64encode(buffer.getvalue()).decode()

    # Calculate scaled dimensions
    height, width = processed_frames[0].shape[:2]
    scaled_width, scaled_height = int(width * scale), int(height * scale)

    # Return HTML video tag
    return HTML(f'''
        <video width="{scaled_width}" height="{scaled_height}" controls autoplay loop muted>
            <source src="data:video/{format};base64,{video_data}" type="video/{format}">
            Your browser does not support the video tag.
        </video>
    ''')


def render_videos(video_tuples, **kwargs):
    from IPython.display import HTML

    videos_frames = [x[0] for x in video_tuples] # Extract video frame lists from tuples
    videos = [x if isinstance(x, HTML) else render_video(x, **kwargs) for x in videos_frames] # Render videos if necessary
    labels = [x[1] for x in video_tuples] # Extract labels from tuples
    cell_style = "border: 2px solid grey; text-align: center; padding: 5px; font-weight: bold; text-transform: uppercase;" # The CSS for each cell
    video_row = "<tr>" + "".join([f"<td style='{cell_style}'>{video.data}</td>" for video in videos]) + "</tr>" # The row that shows the video
    label_row = "<tr>" + "".join([f"<td style='{cell_style}'>{label}</td>" for label in labels]) + "</tr>" # The row that shows the video's label
    return HTML(f"""
    <table style="width:100%; border-collapse: collapse; text-align: center;">
        {video_row}
        {label_row}
    </table>
    """)


def render_video_from_dir(
    directory_path,      
    scale=1,             
    fps=30,              
    format='mp4'       
):
    import os
    import re
    import cv2
    import base64
    import tempfile
    from IPython.display import HTML
    import imageio_ffmpeg as ffmpeg

    def get_numeric_sort_key(filename):
        return int(re.search(r'\d+', os.path.splitext(filename)[0]).group())

    def adjust_frame_size(frame, block_size=16):
        h, w = frame.shape[:2]
        new_h = ((h + block_size - 1) // block_size) * block_size
        new_w = ((w + block_size - 1) // block_size) * block_size
        return cv2.resize(frame, (new_w, new_h))

    # Collect and sort image files
    image_files = sorted([
        f for f in os.listdir(directory_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ], key=get_numeric_sort_key)

    if not image_files:
        raise ValueError("No image files found in the directory.")

    first_frame = cv2.imread(os.path.join(directory_path, image_files[0]))
    if first_frame is None:
        raise ValueError("First image couldn't be read.")
    
    first_frame = adjust_frame_size(first_frame)
    height, width = first_frame.shape[:2]
    scaled_width, scaled_height = int(width * scale), int(height * scale)

    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmpfile:
        output_path = tmpfile.name

    # --- Correct way to use the generator ---
    writer = ffmpeg.write_frames(
        output_path,
        size=(width, height),
        fps=fps,
        codec='libx264'
    )
    next(writer)  # Start the generator

    try:
        for fname in image_files:
            img_path = os.path.join(directory_path, fname)
            frame = cv2.imread(img_path)
            #if frame is None: continue
            frame = adjust_frame_size(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.send(frame.tobytes())
    finally:
        try:
            writer.close()
        except StopIteration:
            pass

    # Read video and encode to base64
    with open(output_path, "rb") as f:
        video_data = base64.b64encode(f.read()).decode()

    os.remove(output_path)

    return HTML(f'''
        <video width="{scaled_width}" height="{scaled_height}" controls autoplay loop muted>
            <source src="data:video/{format};base64,{video_data}" type="video/{format}">
            Your browser does not support the video tag.
        </video>
    ''')


def save_frames_to_dir(iterator, output_dir, ext="jpg", frame_offset=0, show_progress=True):
    import os
    import torch
    from torchvision.transforms import ToPILImage
    from tqdm import tqdm

    """
    Dumps tensors from an iterator to numbered image files in the given directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    to_pil = ToPILImage()

    if show_progress: iterator = tqdm(iterator, desc="Saving frames")

    for i, frame in enumerate(iterator):
        if isinstance(frame, torch.Tensor):
            frame = to_pil(frame)
        frame_id = frame_offset + i
        filename = os.path.join(output_dir, f"{frame_id}.{ext}")
        frame.save(filename)


def render_loader_video(loader, separator_width=10, separator_color=0):
    import torch
    import tempfile
    from tqdm import tqdm

    frame_offset = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        for x, y in tqdm(loader, desc="Rendering frames"):
            B, C, H, W = x.shape

            # Create black separator bar
            bar = torch.full((B, C, H, separator_width), fill_value=separator_color, device=x.device)

            # Concatenate input | separator | target
            side_by_side = torch.cat([x, bar, y], dim=3)  # Concatenate along width

            image_tensors = side_by_side.cpu().unbind(0)  # Unbind to list of individual images

            save_frames_to_dir(image_tensors, temp_dir, frame_offset=frame_offset, show_progress=False)
            frame_offset += len(image_tensors)

        video = render_video_from_dir(temp_dir)
    return video


def render_autoencoder_video(model, loader, compare_inputs=True, separator_width=10, separator_color=0):
    """
    Renders a video of model predictions. If compare_inputs is True, renders side-by-side input and output
    with an optional separator bar.

    Args:
        model: PyTorch model.
        loader: DataLoader returning (x, y) batches.
        compare_inputs (bool): If True, shows input | separator | prediction.
        separator_width (int): Width of vertical separator in pixels.
        separator_color (float or Tensor): Value(s) for separator color. Use float for grayscale, Tensor of shape (C,) for RGB.
    """

    import torch
    import tempfile
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device
    frame_offset = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for x, _ in tqdm(loader):
            x = x.to(device)
            with torch.no_grad():
                y_pred = model(x)

            if compare_inputs:
                B, C, H, W = x.shape

                # Create separator bar
                if isinstance(separator_color, torch.Tensor):
                    if separator_color.shape != (C,):
                        raise ValueError(f"separator_color tensor must have shape ({C},), got {separator_color.shape}")
                    bar = separator_color.view(C, 1, 1).expand(C, H, separator_width)
                else:
                    bar = torch.full((C, H, separator_width), fill_value=separator_color, device=device)

                bar = bar.unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, H, separator_width)

                # Concatenate [input | separator | prediction]
                side_by_side = torch.cat([x, bar, y_pred], dim=3)
                image_tensors = side_by_side.unbind(0)
            else:
                image_tensors = y_pred.unbind(0)

            save_frames_to_dir(image_tensors, temp_dir, frame_offset=frame_offset, show_progress=False)
            frame_offset += len(image_tensors)

        video = render_video_from_dir(temp_dir)
        return video
