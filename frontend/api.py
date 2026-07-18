import requests

from config import API_URL


def _request(method, path, **kwargs):
    try:
        response = requests.request(
            method, f"{API_URL}{path}", timeout=30, **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"detail": f"API request failed: {exc}"}


def chat(question):
    return _request("POST", "/chat", json={"question": question})


def upload(file):
    return _request("POST", "/upload", files={"file": file})


def stats():
    return _request("GET", "/stats")


def knowledge():
    return _request("GET", "/knowledge-base")


def documents():
    return _request("GET", "/documents")
