def disconnect_after_timeout(timeout_seconds=None):    
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
    from .popdesk import notify
    
    notebook_id = os.getenv("NOTEBOOK_ID"); assert notebook_id is not None, "NOTEBOOK_ID environment variable is not set"
    notification_url = os.getenv("NOTIFICATION_URL"); assert notification_url is not None, "NOTIFICATION_URL environment variable is not set"
    notification_auth_token = os.getenv("NOTIFICATION_AUTH_TOKEN"); assert notification_auth_token is not None, "NOTIFICATION_AUTH_TOKEN environment variable is not set"
    
    # Send notification
    notify(
        notification_url,
        notification_auth_token,
        notebook_id,
        message
    )

    # Disconnect runtime after timeout
    disconnect_after_timeout(timeout_seconds=timeout_seconds)

def load_secrets_into_env(keys):
    import os
    from google.colab import userdata

    for key in keys:
        # Load the secret from userdata
        value = userdata.get(key)
        
        # Assert that the value is available
        assert value, f"Key {key} not found in userdata"

        # Set the secret in the environment
        os.environ[key] = value