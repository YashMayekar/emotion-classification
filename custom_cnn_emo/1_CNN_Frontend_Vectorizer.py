import vocabAsset from './tokenizer_vocab.json';

interface TokenizerVocab {
    [key: string]: number;
}

const vocab: TokenizerVocab = vocabAsset;
const MAX_SEQUENCE_LENGTH = 50;

/**
 * Sanitizes and vectorizes a raw text string input into a token array.
 * @param text The transcript snippet from the S2S system.
 * @returns Int32Array formatted for direct ingestion into your ONNX model tensor.
 */
export function tokenizeTranscript(text: string): Int32Array {
    // Initialize an empty vector array filled with 0 (<PAD> ID)
    const tensorBuffer = new Int32Array(MAX_SEQUENCE_LENGTH);
    
    // Convert to lowercase and clean up illegal or non-mapped characters
    const cleanText = text
        .toLowerCase()
        .replace(/[^a-zA-Z\u0900-\u097F\s.,!?]/g, "")
        .replace(/\s+/g, " ")
        .trim();

    // Map characters to structural IDs step-by-step
    const characterList = cleanText.split('');
    const iterations = Math.min(characterList.length, MAX_SEQUENCE_LENGTH);

    for (let i = 0; i < iterations; i++) {
        const char = characterList[i];
        
        if (char in vocab) {
            tensorBuffer[i] = vocab[char];
        } else {
            tensorBuffer[i] = vocab["<UNK>"]; // Fallback for unmapped unique tokens
        }
    }

    return tensorBuffer;
}

// Example usage within your audio stream callback hook:
// const inputIds = tokenizeTranscript("Mujhe syntax samajh nahi aa raha!");
// console.log(inputIds); // Output: Int32Array(50) [ 15, 23, 12, ... 0, 0, 0 ]
