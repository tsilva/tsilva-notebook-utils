def github_get_file_contents(repo, file_path, branch="main", owner=None, token=None):
    import os
    import requests
    import base64

    if owner is None: owner = os.getenv("GITHUB_OWNER")
    if not owner: raise ValueError("GITHUB_OWNER environment variable not set")

    if token is None: token = os.getenv("GITHUB_TOKEN")
    if not token: raise ValueError("GITHUB_TOKEN environment variable not set")

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    content = response.json()
    return base64.b64decode(content["content"]).decode("utf-8")
