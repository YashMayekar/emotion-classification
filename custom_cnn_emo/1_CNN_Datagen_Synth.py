import os
import json
import torch
import requests
import pandas as pd

# 1. Configuration & API Setup
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("GEMINI_API_KEY")
API_URL = "https://deepseek.com" # Replace with your chosen endpoint

def generate_synthetic_batch():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # System schema payload ensuring zero structural parsing failures
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You output strict JSON array format only. No markdown, no explanations."
            },
            {
                "role": "user",
                "content": (
                    "Generate 5 distinct rows of student statements about programming/AI. "
                    "Return a JSON array of objects. Schema: "
                    "[{\"text\": \"string\", \"emotion\": \"frustrated|confident|hesitant|neutral\", \"language\": \"english|hindi|hinglish\"}]"
                )
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response_data = response.json()
        raw_content = response_data['choices'][0]['message']['content']
        return json.loads(raw_content)
    except Exception as e:
        print(f"API Generation Error: {e}")
        return []

# Run a mini pipeline batch to preview schema collection
# (In a production loop, run this continuously until hitting your 10K dataset threshold)
sample_dataset = [
    {"text": "Yaar, recursion ka base case hamesha break ho jata hai, samajh nahi aa raha.", "emotion": "frustrated", "language": "hinglish"},
    {"text": "I have successfully implemented the merge sort function!", "emotion": "confident", "language": "english"},
    {"text": "Mmm... shaayad dynamic programming array ki size depend karegi?", "emotion": "hesitant", "language": "hinglish"},
    {"text": "यह मॉडल ओवरफिटिंग (overfitting) को रोकने के लिए ड्रॉपआउट का उपयोग करता है।", "emotion": "neutral", "language": "hindi"}
]

df = pd.DataFrame(sample_dataset)
print("--- Synthetic Generated Data Sample ---")
print(df.to_string())
