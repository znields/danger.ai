import requests

url = "https://us-central1-arched-glow-229104.cloudfunctions.net/dangerScores"

payload = {
	"name": "",
	"gcp": [],
	"pose": [],
	"pixel": []
}

headers = {
	'Content-Type': "application/json",
}

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)
