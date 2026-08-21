import json
import re
from collections import Counter

def build_and_save_vocab_asset(dataset_texts, save_path="tokenizer_vocab.json"):
    # 1. Base tokens mandatory for sequence padding and OOV boundaries
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    # 2. Extract and sanitize clean character matrices
    combined_raw_text = " ".join(dataset_texts).lower()
    # Keep alphanumeric characters, Devanagari script blocks, and basic punctuation
    sanitized_text = re.sub(r"[^a-zA-Z\u0900-\u097F\s.,!?]", "", combined_raw_text)
    
    # 3. Sort by frequency to prioritize common structures if clipping vocab size
    char_counts = Counter(sanitized_text)
    
    for char, _ in char_counts.most_common():
        if char not in vocab:
            vocab[char] = len(vocab)
            
    # 4. Write pure serialization asset out to directory folder
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    print(f"[Asset Created] Tokenizer vocabulary successfully written to: {save_path}")
    print(f"Total Unique Vocabulary Size: {len(vocab)} tokens.")
    return vocab

# Mock Execution Run
mock_dataset = [
    "Sir code me error aa raha hai", 
    "मुझे रिकर्सन समझ नहीं आ रहा।"
]
vocab_asset = build_and_save_vocab_asset(mock_dataset)
