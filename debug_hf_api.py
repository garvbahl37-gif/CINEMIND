import os
import requests
from dotenv import load_dotenv

load_dotenv(".env")
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
payload = {
    "inputs": "[INST] Who acts in 3 Idiots? [/INST]",
    "parameters": {"max_new_tokens": 50}
}

urls = [
    "https://api-inference.huggingface.co/models/gpt2",
    "https://router.huggingface.co/models/gpt2",
    "https://router.huggingface.co/gpt2",
]

url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
print(f"Testing: {url}")
try:
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
