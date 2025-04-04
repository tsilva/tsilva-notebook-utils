def login(token: str):
    from huggingface_hub import login, whoami
    login(token=token)
    return whoami()
