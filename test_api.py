import requests

response = requests.post("http://127.0.0.1:5000/predict", json={"text": "I feel anxious about tomorrow.", "model": "bert"})
print(response.json())
