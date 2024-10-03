import requests

# API endpoint
url = 'http://localhost:5000/predict'

# Data payload
data = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

# Send POST request
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)