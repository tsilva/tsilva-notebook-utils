def disconnect_after_timeout(timeout_seconds=None): 
    # Skip if not in Google Colab
    try: import google.colab
    except: return

    import time
    from tqdm import tqdm
    from google.colab import runtime

      # Default to 5 minutes if not specified
    if timeout_seconds is None: timeout_seconds = 60 * 5

    print(f"Starting idle timeout check. Will disconnect after {timeout_seconds} seconds of no interruption...")

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
    import os
    from .notifications import send_popdesk_notification
    
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
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    try:
        from google.colab import userdata
        for key in keys:
            value = userdata.get(key)
            assert value, f"Key {key} not found in userdata"
            os.environ[key] = value
    except:
        from dotenv import load_dotenv
        load_dotenv(override=True)

    values = []
    for key in keys:
        value = os.getenv(key)
        assert value, f"Key {key} not found in environment variables"
        values.append(value)


def notebook_id_from_title():
    # Skip if not in Google Colab
    try: import google.colab
    except: return

    import re
    import unicodedata
    from google.colab import _message

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
