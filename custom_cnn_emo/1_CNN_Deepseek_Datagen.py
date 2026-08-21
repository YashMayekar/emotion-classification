import os
import json
import requests

def generate_with_deepseek(batch_size: int = 10):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    url = "https://deepseek.com"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system", 
                "content": (
                    "You are a dataset generator. Output strict JSON only. Your response must be an object with a single key 'dataset' containing an array of objects. "
                    "Each object must match this schema exactly: {\"text\": \"string\", \"emotion\": \"frustrated|confident|hesitant|neutral\", \"language\": \"english|hindi|hinglish\"}"
                )
            },
            {
                "role": "user", 
                "content": f"Generate {batch_size} unique rows of coding student statements in English, Hindi, and Hinglish."
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['choices'][0]['message']['content']

# print(generate_with_deepseek(5))
