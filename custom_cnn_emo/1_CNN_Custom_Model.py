class EmotionDataset(Dataset):
    def __init__(self, data, vocab, label_map, max_len):
        self.data = data
        self.vocab = vocab
        self.label_map = label_map
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text, label = self.data[idx]
        x = vectorize_text(text, self.vocab, self.max_len)
        y = torch.tensor(self.label_map[label], dtype=torch.long)
        return x, y

# Instantiate DataLoader
dataset = EmotionDataset(cleaned_data, vocab, label_map, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
