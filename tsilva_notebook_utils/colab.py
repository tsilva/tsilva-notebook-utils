import os
import re
import time
import unicodedata



try:  # noqa: SIM105
    import google.colab  # type: ignore
    from google.colab import _message, runtime, userdata  # type: ignore
    IN_COLAB = True
except Exception:  # noqa: BLE001
    IN_COLAB = False
    _message = None
    runtime = None
    userdata = None

from .notifications import send_popdesk_notification


def disconnect_after_timeout(timeout_seconds=None):
    if not IN_COLAB:
        return

      # Default to 5 minutes if not specified
    if timeout_seconds is None: timeout_seconds = 60 * 5

    print(f"Starting idle timeout check. Will disconnect after {timeout_seconds} seconds of no interruption...")

    from tqdm import tqdm

    start_time = time.time()
    with tqdm(total=timeout_seconds, desc="Idle Timeout", unit="s") as pbar:
        while time.time() - start_time < timeout_seconds:
            elapsed = int(time.time() - start_time)
            pbar.n = elapsed
            pbar.refresh()
            # Add logic here to detect activity and reset start_time if needed
            time.sleep(1)

    print("Idle timeout reached. Disconnecting runtime...")
    runtime.unassign()

def notify_and_disconnect_after_timeout(message="Notebook execution finished!", timeout_seconds=None):
    
    
    notebook_id = os.getenv("NOTEBOOK_ID"); assert notebook_id is not None, "NOTEBOOK_ID environment variable is not set"
    notification_url = os.getenv("NOTIFICATION_URL"); assert notification_url is not None, "NOTIFICATION_URL environment variable is not set"
    notification_auth_token = os.getenv("NOTIFICATION_AUTH_TOKEN"); assert notification_auth_token is not None, "NOTIFICATION_AUTH_TOKEN environment variable is not set"
    
    # Send notification
    send_popdesk_notification(
        notification_url,
        notification_auth_token,
        notebook_id,
        message
    )

    # Disconnect runtime after timeout
    disconnect_after_timeout(timeout_seconds=timeout_seconds)


def load_secrets_into_env(keys):
    from dotenv import load_dotenv

    load_dotenv(override=True)

    try:
        for key in keys:
            value = userdata.get(key)
            assert value, f"Key {key} not found in userdata"
            os.environ[key] = value
    except Exception:
        load_dotenv(override=True)

    values = []
    for key in keys:
        value = os.getenv(key)
        assert value, f"Key {key} not found in environment variables"
        values.append(value)


def notebook_id_from_title():
    if not IN_COLAB:
        return


    def _slugify(value):
        value = str(value)
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^a-zA-Z0-9]+', '-', value)
        value = value.strip('-')
        return value.lower()
        
    metadata = _message.blocking_request('get_ipynb')
    text = metadata['ipynb']['cells'][0]['source'][0]
    slug = _slugify(text)
    return slug
