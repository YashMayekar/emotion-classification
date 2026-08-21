# Create a character/sub-word map to inherently support cross-lingual variations
all_text = " ".join([item[0] for item in cleaned_data])
tokens = list(all_text) # Character-level chosen for robust Hinglish processing

vocab = {"<PAD>": 0, "<UNK>": 1}
token_counts = Counter(tokens)
for token, count in token_counts.items():
    if count >= 1: # Lower threshold to catch spelling variations
        vocab[token] = len(vocab)

# Mapping Labels
label_map = {"happy": 0, "sad": 1, "angry": 2, "neutral": 3}
num_classes = len(label_map)

MAX_LEN = 50

def vectorize_text(text, vocab, max_len):
    vector = [vocab.get(char, vocab["<UNK>"]) for char in list(text)]
    if len(vector) < max_len:
        vector += [vocab["<PAD>"]] * (max_len - len(vector))
    else:
        vector = vector[:max_len]
    return torch.tensor(vector, dtype=torch.long)
