
import requests


def send_popdesk_notification(url, auth_token, title, message):
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    data = {
        "title": title,
        "message": message
    }
    requests.post(url, json=data, headers=headers)
