def disconnect_after_timeout(timeout_seconds=300):    
    import time
    from google.colab import runtime
    from tqdm import tqdm

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
