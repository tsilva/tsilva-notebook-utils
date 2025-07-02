import base64
import os



def github_get_file_contents(repo, file_path, branch="main", username=None, token=None):
    import requests

    if username is None: username = os.getenv("GITHUB_USERNAME")
    if not username: raise ValueError("GITHUB_USERNAME environment variable not set")

    if token is None: token = os.getenv("GITHUB_TOKEN")
    if not token: raise ValueError("GITHUB_TOKEN environment variable not set")

    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{file_path}?ref={branch}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    content = response.json()
    return base64.b64decode(content["content"]).decode("utf-8")
