import torch.onnx

# Switch model to evaluation mode
model.eval()

# Dummy input matching the MAX_LEN constraint (Batch Size 1, Sequence Length 50)
dummy_input = torch.zeros((1, 50), dtype=torch.long)

# Export the 1D-CNN architecture to an ONNX graph
torch.onnx.export(
    model, 
    dummy_input, 
    "multilingual_emotion_cnn.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input_text_ids'],
    output_names=['emotion_logits'],
    dynamic_axes={'input_text_ids': {0: 'batch_size'}, 'emotion_logits': {0: 'batch_size'}}
)
