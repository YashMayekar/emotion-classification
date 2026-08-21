import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Define the targeted validation schema
class EmotionDataRow(BaseModel):
    text: str = Field(description="The student statement in English, Hindi, or Hinglish")
    emotion: str = Field(description="Must be one of: frustrated, confident, hesitant, neutral")
    language: str = Field(description="Must be one of: english, hindi, hinglish")

class SyntheticDataset(BaseModel):
    dataset: list[EmotionDataRow]

def generate_with_gemini(batch_size: int = 10):
    # Initializes client utilizing GEMINI_API_KEY environment variable implicitly
    client = genai.Client()
    
    prompt = f"Generate a unique list of {batch_size} distinct statements a student might tell an AI technical interviewer."
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are an AI data generator for K-12 and undergrad programming students in India.",
            response_mime_type="application/json",
            response_schema=SyntheticDataset,
            temperature=0.7,
        ),
    )
    
    # Securely parsed straight into a structured JSON string dictionary 
    return response.text

# print(generate_with_gemini(5))
