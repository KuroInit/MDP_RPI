import requests
server_url = "http://<Server IP>:5000/capture"
try:
    response = requests.get(server_url, timeout=5)  
    data = response.json()
    print("Result:", data)

except Exception as e:
    print("Request failed:", e)
