import cv2
import base64
import imageio
from io import BytesIO
from IPython.display import HTML

def render_video(
    frames,              # List of frames to render. Each frame can be an image or a tuple (image, label)
    scale=1,             # Scale factor for the video dimensions
    fps=30,              # Frames per second for the video
    format='mp4',        # Video format
    font_scale=0.5       # Font scale for text shown in video corner
):
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
