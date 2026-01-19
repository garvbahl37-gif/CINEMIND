import requests

url = "https://bharatverse11-movie-recommender-system.hf.space/chat/message"
# Simulate an OPTIONS preflight request or just a POST to see headers
try:
    print(f"Checking OPTIONS for {url}...")
    response = requests.options(url, headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST"
    })
    print(f"Status Code: {response.status_code}")
    print("Headers:")
    for k, v in response.headers.items():
        if "Access-Control" in k:
            print(f"  {k}: {v}")
            
    print("\nChecking POST response headers...")
    response_post = requests.post(url, json={"message": "hi", "session_id": "test"})
    for k, v in response_post.headers.items():
        if "Access-Control" in k:
            print(f"  {k}: {v}")

except Exception as e:
    print(f"Request failed: {e}")
