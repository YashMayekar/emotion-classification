import * as ort from 'onnxruntime-web';

// Simple Character Map Vectorizer matching your python token configuration
function tokenizeText(text: string, maxLen: number = 50): Int32Array {
    // Basic fallback map example placeholder configuration
    const mockVocab: Record<string, number> = { "<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3, "e": 4, "h": 5, "m": 6, "r": 7 }; 
    const vector = new Int32Array(maxLen); // Zero initialized implicitly acts as <PAD>
    
    const chars = text.toLowerCase().split('');
    for (let i = 0; i < Math.min(chars.length, maxLen); i++) {
        vector[i] = mockVocab[chars[i]] ?? mockVocab["<UNK>"];
    }
    return vector;
}

export async function runWebInference(rawTranscript: string): Promise<string> {
    const emotions = ["frustrated", "confident", "hesitant", "neutral"];
    
    // 1. Instantiate the WASM runtime session
    const session = await ort.InferenceSession.create('/models/optimized_emotion_classifier.onnx');
    
    // 2. Vectorize inputs into standard typed arrays
    const sequenceLength = 50;
    const tokenizedData = tokenizeText(rawTranscript, sequenceLength);
    
    // 3. Construct input tensor mapped into [BatchSize: 1, SequenceLength: 50]
    const inputTensor = new ort.Tensor('int32', tokenizedData, [1, sequenceLength]);
    
    // 4. Fire localized underlying hardware compilation thread execution
    const feeds = { input_ids: inputTensor };
    const results = await session.run(feeds);
    
    // 5. Interpret output logits array tensor indices
    const logits = results.logits.data as Float32Array;
    const maxIdx = logits.reduce((maxI, el, i, arr) => el > arr[maxI] ? i : maxI, 0);
    
    return emotions[maxIdx];
}
