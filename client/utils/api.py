# frontend/utils/api.py
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

def upload_policy_api(files):
    files_payload=[("files",(f.name,f.read(),"application/pdf")) for f in files]
    return requests.post(f"{API_URL}/upload_policy/", files=files_payload)

def moderate_api(current_file):
    files = {"current_file": (current_file.name, current_file.read(), "application/pdf")}
    return requests.post(f"{API_URL}/moderate/", files=files)
