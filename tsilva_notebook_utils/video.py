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


def render_loader_video(loader, input_key="input", target_key="target", separator_width=10, separator_color=0, **render_kwargs):
    import torch
    import tempfile
    from tqdm import tqdm

    frame_offset = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        for batch in tqdm(loader, desc="Rendering frames"):
            x, y = batch[input_key], batch[target_key]
            B, C, H, W = x.shape

            # Create black separator bar
            bar = torch.full((B, C, H, separator_width), fill_value=separator_color, device=x.device)

            # Concatenate input | separator | target
            side_by_side = torch.cat([x, bar, y], dim=3)  # Concatenate along width

            image_tensors = side_by_side.cpu().unbind(0)  # Unbind to list of individual images

            save_frames_to_dir(image_tensors, temp_dir, frame_offset=frame_offset, show_progress=False)
            frame_offset += len(image_tensors)

        video = render_video_from_dir(temp_dir, **render_kwargs)
    return video


def render_autoencoder_video(model, loader, input_key="input", target_key="target", separator_width=10, separator_color=0, **render_kwargs):
    """
    Renders a video comparing model input, prediction, and ground truth target side-by-side.
    The layout for each frame is: [input | separator | prediction | separator | target].

    Args:
        model: PyTorch model.
        loader: DataLoader returning batches. Each batch must be a dictionary
                containing keys specified by `input_key` and `target_key`.
                Assumes tensors are in (B, C, H, W) format.
        input_key (str): Dictionary key for the input tensor in the batch.
        target_key (str): Dictionary key for the target (ground truth) tensor
                          in the batch.
        separator_width (int): Width of the vertical separator bars in pixels.
        separator_color (float or Tensor): Value(s) for separator color. Use float for grayscale,
                                           Tensor of shape (C,) for RGB.
        fps (int): Frames per second for the output video.
    """
    import torch
    import tempfile
    from tqdm import tqdm
    # Make sure the necessary helper functions are available in the scope
    # from tsilva_notebook_utils.video import render_video_from_dir, save_frames_to_dir

    model.eval()
    # Determine device from model parameters or fallback
    try:
        device = next(model.parameters()).device
    except StopIteration:
        print("Warning: Could not determine device from model parameters. Assuming CPU or device from first batch.")
        try:
            first_batch = next(iter(loader))
            if input_key not in first_batch:
                 raise KeyError(f"Input key '{input_key}' not found in the first batch.")
            device = first_batch[input_key].device
            print(f"Inferred device: {device}")
        except Exception as e:
            print(f"Could not infer device from data loader: {e}. Defaulting to CPU.")
            device = torch.device('cpu')
            if isinstance(model, torch.nn.Module):
                model.to(device)

    frame_offset = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Saving temporary frames to: {temp_dir}")
        for batch in tqdm(loader, desc="Processing batches"):
            # Ensure batch is a dictionary and contains both input and target keys
            if not isinstance(batch, dict) or input_key not in batch or target_key not in batch:
                 print(f"Warning: Batch format incorrect or missing key ('{input_key}' or '{target_key}'). Skipping batch.")
                 continue

            x_batch = batch[input_key].to(device)
            target_batch = batch[target_key].to(device)

            # Check if input and target shapes match (important for visual comparison)
            if x_batch.shape != target_batch.shape:
                 print(f"Warning: Input shape {x_batch.shape} differs from target shape {target_batch.shape}. Skipping batch.")
                 continue

            with torch.no_grad():
                try:
                    result = model(x_batch)
                    if isinstance(result, tuple):
                        y_pred = result[0]
                    else:
                        y_pred = result
                except Exception as e:
                    print(f"Error during model inference: {e}. Skipping batch.")
                    continue

            # Ensure prediction has the same dimensions as input/target B, C, H, W
            if y_pred.shape != x_batch.shape:
                 print(f"Warning: Prediction shape {y_pred.shape} differs from input/target shape {x_batch.shape}. Skipping batch.")
                 continue

            B, C, H, W = x_batch.shape
            frames_to_save = []

            # Create separator bar (dtype will be matched later)
            if isinstance(separator_color, torch.Tensor):
                if separator_color.numel() == 1:
                     separator_color_val = separator_color.item()
                     # Create bar with a base dtype, will convert later if needed
                     bar_prototype = torch.full((C, H, separator_width), fill_value=separator_color_val, device=device)
                elif separator_color.shape == (C,):
                     # Ensure separator tensor is on the correct device
                     bar_prototype = separator_color.to(device).view(C, 1, 1).expand(C, H, separator_width)
                else:
                     raise ValueError(f"separator_color tensor must have shape ({C},) or be a single element, got {separator_color.shape}")
            else: # Assume float or int
                 bar_prototype = torch.full((C, H, separator_width), fill_value=separator_color, device=device)

            # Determine the target dtype (use prediction's dtype as reference)
            target_dtype = y_pred.dtype

            # Ensure all parts have the same dtype before concatenation
            x_batch_compat = x_batch.to(dtype=target_dtype)
            target_batch_compat = target_batch.to(dtype=target_dtype)
            y_pred_compat = y_pred.to(dtype=target_dtype) # Already correct dtype, but explicit is fine
            bar = bar_prototype.to(dtype=target_dtype) # Convert bar to target dtype
            bar = bar.unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, H, separator_width)


            # Concatenate [input | separator | prediction | separator | target]
            combined_frame = torch.cat([x_batch_compat, bar, y_pred_compat, bar, target_batch_compat], dim=3)
            frames_to_save = combined_frame.unbind(0)

            # Save the frames for the current batch
            try:
                 # Ensure save_frames_to_dir is defined and imported
                 save_frames_to_dir(frames_to_save, temp_dir, frame_offset=frame_offset, show_progress=False)
                 frame_offset += len(frames_to_save)
            except NameError:
                 print("Error: 'save_frames_to_dir' function is not defined. Please import or define it.")
                 return None # Stop execution if helper is missing
            except Exception as e:
                 print(f"Error saving frames: {e}")
                 # Decide if you want to stop or continue
                 # return None

        # Render the video from the saved frames
        if frame_offset == 0:
             print("No frames were generated. Cannot render video.")
             return None

        print(f"Rendering video from {frame_offset} frames...")
        try:
            # Ensure render_video_from_dir is defined and imported
            video = render_video_from_dir(temp_dir, **render_kwargs)
            print("Video rendering complete.")
            return video
        except NameError:
            print("Error: 'render_video_from_dir' function is not defined. Please import or define it.")
            return None # Stop execution if helper is missing
        except Exception as e:
            print(f"Error rendering video: {e}")
            return None


def render_video_from_frames(frames_t, **render_kwargs):
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        save_frames_to_dir(frames_t, temp_dir)
        video = render_video_from_dir(temp_dir, **render_kwargs)
    return video
