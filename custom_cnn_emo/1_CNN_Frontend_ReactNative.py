import React, { useEffect } from 'react';
import { Tensor, InferenceSession } from 'onnxruntime-react-native';

const EMOTIONS = ["frustrated", "confident", "hesitant", "neutral"];
const MAX_LEN = 50;

// Reusable vectorization mechanism for mobile execution bounds
const textToInputIds = (text: string): Int32Array => {
    const vector = new Int32Array(MAX_LEN);
    // Populate using your absolute production vocab indexing file asset structure
    return vector;
};

export const useMobileEmotionClassifier = (modelAssetPath: string) => {
    let session: InferenceSession | null = null;

    useEffect(() => {
        const initModel = async () => {
            // Path configuration targets local bundle build structures seamlessly
            session = await InferenceSession.create(modelAssetPath);
        };
        initModel();
    }, [modelAssetPath]);

    const classifyStudentEmotion = async (textTranscript: string): Promise<string> => {
        if (!session) throw new Error("ONNX Engine not initialized yet.");

        const inputIdsData = textToInputIds(textTranscript);
        // Create input tensor explicit bounds
        const inputTensor = new Tensor('int32', inputIdsData, [1, MAX_LEN]);
        
        // Execute inference using mobile optimized architecture
        const outputs = await session.run({ input_ids: inputTensor });
        const outputLogits = outputs.logits.data as Float32Array;
        
        // FindArgMax
        let maxIndex = 0;
        let maxVal = outputLogits[0];
        for (let i = 1; i < outputLogits.length; i++) {
            if (outputLogits[i] > maxVal) {
                maxVal = outputLogits[i];
                maxIndex = i;
            }
        }
        return EMOTIONS[maxIndex];
    };

    return { classifyStudentEmotion };
};
