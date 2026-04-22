"""Smoke test for a running API (not a pytest)."""

import requests

URL = "http://localhost:8000/predict"
payload = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
}

if __name__ == "__main__":
    response = requests.post(URL, json=payload, timeout=5)
    response.raise_for_status()
    print(response.json())
