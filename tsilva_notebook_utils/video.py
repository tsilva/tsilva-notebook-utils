import os
import re
import cv2
import base64
import torch
import tempfile
from collections.abc import Iterable
from torchvision.transforms import ToPILImage
from IPython.display import HTML
import imageio_ffmpeg as ffmpeg


def save_tensor_frames(frames, output_dir, ext="jpg", start_index=0):
    """
    Saves a list or tensor of image frames to a specified directory.

    Args:
        frames: A list or tensor of shape [T, C, H, W] or [C, H, W].
        output_dir: Directory where frames will be saved.
        ext: Image extension.
        start_index: Starting index for naming the saved frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    to_pil = ToPILImage()

    for i, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = to_pil(frame)
        frame.save(os.path.join(output_dir, f"{start_index + i}.{ext}"))


def create_video_from_frames(directory_path, scale=1, fps=30, format='mp4'):
    """
    Creates a video from a directory of image frames.

    Args:
        directory_path: Path to directory with image frames.
        scale: Scale factor for output video resolution.
        fps: Frames per second.
        format: Video file format.

    Returns:
        HTML object to display the embedded video.
    """
    image_files = sorted([
        f for f in os.listdir(directory_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ], key=lambda x: int(re.search(r'\d+', os.path.splitext(x)[0]).group()))

    if not image_files:
        raise ValueError("No image files found in the directory.")

    first_frame = cv2.imread(os.path.join(directory_path, image_files[0]))
    height, width = first_frame.shape[:2]
    scaled_width, scaled_height = int(width * scale), int(height * scale)

    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmpfile:
        output_path = tmpfile.name

    writer = ffmpeg.write_frames(
        output_path,
        size=(width, height),
        fps=fps,
        codec='libx264'
    )
    next(writer)

    try:
        for fname in image_files:
            frame = cv2.imread(os.path.join(directory_path, fname))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.send(frame.tobytes())
    finally:
        try:
            writer.close()
        except StopIteration:
            pass

    with open(output_path, "rb") as f:
        video_data = base64.b64encode(f.read()).decode()
    os.remove(output_path)

    return HTML(f'''
        <video width="{scaled_width}" height="{scaled_height}" controls autoplay loop muted>
            <source src="data:video/{format};base64,{video_data}" type="video/{format}">
            Your browser does not support the video tag.
        </video>
    ''')


def render_video_from_batches(data, frame_mapper=None, **video_kwargs):
    """
    Renders a video from either a list/tensor of frames or batches processed via frame_mapper.

    Args:
        data: Tensor, list of frames, or iterable of batches.
        frame_mapper: Optional function that processes a batch into a list of frames.
        video_kwargs: Passed to create_video_from_frames.

    Returns:
        HTML object for embedded video.
    """
    frame_index = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        if isinstance(data, torch.Tensor):
            save_tensor_frames(data, temp_dir)

        elif isinstance(data, list) and (not frame_mapper or isinstance(data[0], torch.Tensor)):
            save_tensor_frames(data, temp_dir)

        else:
            for batch in data:
                frames = frame_mapper(batch)
                save_tensor_frames(frames, temp_dir, start_index=frame_index)
                frame_index += len(frames)

        return create_video_from_frames(temp_dir, **video_kwargs)


def render_video_from_dataloader(loader, input_key="input", target_key="target",
                                 model=None, separator_width=10, separator_color=0, **video_kwargs):
    """
    Renders model predictions vs targets from a DataLoader as a video.

    Args:
        loader: DataLoader or batch iterator.
        input_key: Key for inputs in the batch.
        target_key: Key for targets in the batch.
        model: Optional model to generate predictions.
        separator_width: Width of the visual separator between images.
        separator_color: Color value for the separator.
        video_kwargs: Passed to create_video_from_frames.

    Returns:
        HTML object of the rendered video.
    """
    def make_frames(batch):
        inputs, targets = batch[input_key], batch[target_key]
        batch_size = inputs.size(0)
        sep = torch.full((batch_size, inputs.size(1), inputs.size(2), separator_width),
                         fill_value=separator_color, device=inputs.device)
        display_stack = [inputs, sep]

        if model is not None:
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                predictions = model(inputs.to(device))
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            predictions = predictions.to(inputs.device)
            display_stack = [inputs, sep, predictions, sep, targets]
        else:
            display_stack.append(targets)

        return torch.cat(display_stack, dim=3).cpu().unbind(0)

    return render_video_from_batches(loader, frame_mapper=make_frames, **video_kwargs)
