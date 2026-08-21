def predict_emotion(text, model, vocab, label_map, max_len):
    model.eval()
    cleaned = clean_multilingual_text(text)
    vectorized = vectorize_text(cleaned, vocab, max_len).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        logits = model(vectorized)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        
    # Invert the dictionary map
    inv_label_map = {v: k for k, v in label_map.items()}
    return inv_label_map[pred_idx.item()], confidence.item()

# Test sentences representing all three variants
test_phrase = "Arre yaar, I lost my wallet again, bohot pareshan hu!"
emotion, score = predict_emotion(test_phrase, model, vocab, label_map, MAX_LEN)
print(f"\nTarget Text: '{test_phrase}'\nPredicted Emotion: {emotion.upper()} (Confidence: {score:.2f})")
