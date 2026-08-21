export class DownstreamStrategyAdapter {
  private emotionHistory: string[] = [];
  private currentActiveEmotion: string = 'neutral';
  private readonly windowSize: number;

  /**
   * @param windowSize The number of continuous text frames needed to trigger a behavioral shift.
   */
  constructor(windowSize: number = 3) {
    this.windowSize = windowSize;
  }

  /**
   * Process a new incoming text emotion prediction frame.
   * @param rawEmotion Prediction output string from the 1D-CNN.
   * @returns The stabilized PedagogicalStrategy configuration if a shift happened, otherwise null.
   */
  public processFrame(rawEmotion: string): PedagogicalStrategy | null {
    this.emotionHistory.push(rawEmotion);
    if (this.emotionHistory.length > this.windowSize) {
      this.emotionHistory.shift();
    }

    // Determine the majority emotion in the active processing window
    const counts: Record<string, number> = {};
    let majorityEmotion = this.currentActiveEmotion;
    let maxCount = 0;

    for (const emo of this.emotionHistory) {
      counts[emo] = (counts[emo] || 0) + 1;
      if (counts[emo] > maxCount) {
        maxCount = counts[emo];
        majorityEmotion = emo;
      }
    }

    // Debounce: Only update state if the majority emotion confidently dominates the window
    if (majorityEmotion !== this.currentActiveEmotion && maxCount >= Math.ceil(this.windowSize * 0.66)) {
      this.currentActiveEmotion = majorityEmotion;
      return STRATEGY_MAP[this.currentActiveEmotion];
    }

    return null; // Return null if the state is stable and unchanged
  }

  /**
   * Injects the active strategy modifiers cleanly into the downstream S2S payload array.
   * @param baseLlmMessages Standard system messages framework.
   * @returns Modified payload array prepared for Gemini / DeepSeek streaming ingestion layers.
   */
  public injectStrategyIntoPayload(baseLlmMessages: any[]): any[] {
    const activeStrategy = STRATEGY_MAP[this.currentActiveEmotion];
    
    // Deep copy base message context safely
    const optimizedPayload = JSON.parse(JSON.stringify(baseLlmMessages));
    
    // Prepend or inject the contextual learning modifier instruction securely into the main system loop
    const systemIndex = optimizedPayload.findIndex((m: any) => m.role === 'system');
    if (systemIndex !== -1) {
      optimizedPayload[systemIndex].content += `\n[EMOTIONAL_CONTEXT: ${activeStrategy.systemPromptInjection}]`;
    } else {
      optimizedPayload.unshift({
        role: 'system',
        content: `[EMOTIONAL_CONTEXT: ${activeStrategy.systemPromptInjection}]`
      });
    }
    
    return optimizedPayload;
  }

  public getCurrentEmotion(): string {
    return this.currentActiveEmotion;
  }
}
