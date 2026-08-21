import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 1. Custom Text Cleaner for Code-Mixed Social Data
def clean_multilingual_text(text):
    text = text.lower()
    # Remove URLs, HTML tags, and social media handles
    text = re.sub(r"http\S+|www\S+|<.*?>|@\w+", "", text)
    # Retain English, Devanagari Hindi scripts, common punctuation, and spaces
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Mock Dataset including English, Hindi, and Hinglish elements
raw_data = [
    ("I am feeling absolutely wonderful today!", "happy"),
    ("यह सुनकर मुझे बहुत दुःख हुआ।", "sad"),
    ("Mujhe tumhari ye baat bilkul pasand nahi aayi, bahut gussa aaya.", "angry"),
    ("Let's go celebrate, hum jeet gaye!", "happy"),
    ("Dil toot gaya jab usne mana kar diya.", "sad"),
    ("Main theek hoon, normal chal raha hai sab.", "neutral")
]

cleaned_data = [(clean_multilingual_text(t), e) for t, e in raw_data]
