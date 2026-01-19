import requests
import json

url = "https://bharatverse11-movie-recommender-system.hf.space/chat/message"
payload = {
    "message": "debug hello",
    "session_id": "debug_123"
}
headers = {
    "Content-Type": "application/json"
}

try:
    print(f"Sending POST to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
