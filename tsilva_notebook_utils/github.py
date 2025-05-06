def github_get_file_contents(token, owner, repo, file_path, branch="main"):
    import requests
    import base64

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    content = response.json()
    return base64.b64decode(content["content"]).decode("utf-8")
