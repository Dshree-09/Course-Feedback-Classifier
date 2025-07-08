import pandas as pd
import requests

# Watsonx.ai details
API_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-15"
TOKEN = "eyJraWQiOiIyMDE5MDcyNCIsImFsZyI6IlJTMjU2In0.eyJpYW1faWQiOiJJQk1pZC02OTcwMDBYM0tWIiwiaWQiOiJJQk1pZC02OTcwMDBYM0tWIiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiMGU0Yzk3YWUtNjAwYy00ZWRiLTgxNzQtYWRkNzJlYzM5YTVjIiwiaWRlbnRpZmllciI6IjY5NzAwMFgzS1YiLCJnaXZlbl9uYW1lIjoiQWFyeWFuIiwiZmFtaWx5X25hbWUiOiJTaHJpdmFzdGF2IiwibmFtZSI6IkFhcnlhbiBTaHJpdmFzdGF2IiwiZW1haWwiOiJhYXJ5YW4uc2hyaXZhc3RhdjIwMjNAdml0c3R1ZGVudC5hYy5pbiIsInN1YiI6ImFhcnlhbi5zaHJpdmFzdGF2MjAyM0B2aXRzdHVkZW50LmFjLmluIiwiYXV0aG4iOnsic3ViIjoiYWFyeWFuLnNocml2YXN0YXYyMDIzQHZpdHN0dWRlbnQuYWMuaW4iLCJpYW1faWQiOiJJQk1pZC02OTcwMDBYM0tWIiwibmFtZSI6IkFhcnlhbiBTaHJpdmFzdGF2IiwiZ2l2ZW5fbmFtZSI6IkFhcnlhbiIsImZhbWlseV9uYW1lIjoiU2hyaXZhc3RhdiIsImVtYWlsIjoiYWFyeWFuLnNocml2YXN0YXYyMDIzQHZpdHN0dWRlbnQuYWMuaW4ifSwiYWNjb3VudCI6eyJ2YWxpZCI6dHJ1ZSwiYnNzIjoiOTU3NGJkZWYzYzlmNGQyMWIwMTRiZGQ3NzU5NDRiODQiLCJmcm96ZW4iOnRydWV9LCJpYXQiOjE3NTE4MTcwMTMsImV4cCI6MTc1MTgyMDYxMywiaXNzIjoiaHR0cHM6Ly9pYW0uY2xvdWQuaWJtLmNvbS9pZGVudGl0eSIsImdyYW50X3R5cGUiOiJ1cm46aWJtOnBhcmFtczpvYXV0aDpncmFudC10eXBlOmFwaWtleSIsInNjb3BlIjoiaWJtIG9wZW5pZCIsImNsaWVudF9pZCI6ImRlZmF1bHQiLCJhY3IiOjEsImFtciI6WyJwd2QiXX0.GNEGQdmu69GwHGWsIU8_VzuVBlrMRzVhuaKnI3N2uJMp3l2Cf6Ow8LFk3LU0VBpvxLRzH0IjMrksEy91v4VlovtFFmTTUFbk1SMg5kMl30SdVEceZa0nLfnbTLbwRVcvYbLJBYNS01eVun5jET-0tPx8yBbgsT5yccN5NLFy335UIOGa1Y7OMKNLamOAT1Af-VQeJOTQZ-p-B9kY5XiONQ46Kohq-QFleGjgy9SojOT1kDFMzmiDR_jImSOliBEni_oIkIYTP4CCMxAV9ACFy1fnQ5zPlDjpy7aaOh5_GJ2twFk3UscuAcOTdQia4Ai8LHYJXNmpEp0RgwXmck48vw"  # You should load this from env in real project

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Few-shot examples
FEW_SHOT_PROMPT = """
Classify each feedback into one of the categories: Academics, Facilities, Administration.
Only respond with the category name — nothing else.

Feedback: "There is no drinking water in the building."
Category: Facilities

Feedback: "The admin staff takes too long to respond."
Category: Administration

Feedback: "Professors are not completing the syllabus."
Category: Academics

Feedback: "{}"
Category:"""


def generate_prompt(feedback):
    return FEW_SHOT_PROMPT.format(feedback)
def classify_feedback(feedback):
    prompt = generate_prompt(feedback)
    payload = {
        "model_id": "mistralai/mistral-large",  # or your selected model"mistralai/mistral-medium-2505"

        "input": prompt,
        "parameters": {
            "temperature": 0.3,
            "max_new_tokens": 100
        },
        "project_id":"34c26452-43ca-499f-9a41-0f0f3a9607c8"
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        try:
            raw = response.json()["results"][0]["generated_text"].strip()
            first_line = raw.splitlines()[0].strip()
            print(f"Prompted: {feedback} → Response: {first_line}")
            for label in ["Academics", "Facilities", "Administration"]:
                if label.lower() == first_line.lower():
                    return label
            return "Uncertain"
        except Exception as e:
            print("Error parsing output:", e)
            return "ERROR"
    else:
        print(f"Error {response.status_code}: {response.text}")
        return "ERROR"
# Load your CSV
df = pd.read_csv("feedback.csv")  # Should have a 'Feedback' column

# Classify and write results
df["Predicted_Category"] = df["Feedback"].apply(classify_feedback)
df.to_csv("classified_feedback.csv", index=False)
print("✅ Classification complete. Saved to classified_feedback.csv.")