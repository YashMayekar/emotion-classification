export interface PedagogicalStrategy {
  systemPromptInjection: string; // Dynamic instructions prepended to the LLM context
  voicePitchModifier: number;    // Multiplier for your Piper/TTS model generation speed/pitch
  uiSignal: 'NORMAL' | 'WARN' | 'CELEBRATE' | 'HELP'; // UI Visual indicator states
  interruptionToleranceMs: number; // Tailor how easily the user can interrupt the AI
}

export const STRATEGY_MAP: Record<string, PedagogicalStrategy> = {
  frustrated: {
    systemPromptInjection: "CRITICAL: The student is experiencing friction or confusion. Immediately pause complex technical deep-dives. Break the problem down into a simpler micro-step, provide a gentle hint, or use an analogy.",
    voicePitchModifier: 0.95, // Marginally slower, calmer delivery pacing
    uiSignal: 'HELP',
    interruptionToleranceMs: 200, // Highly sensitive to let a frustrated user break in easily
  },
  hesitant: {
    systemPromptInjection: "NOTICE: The student appears uncertain or hesitant about their response. Do not judge or harshly evaluate. Provide encouraging validation and offer an explicit scaffold or directional question.",
    voicePitchModifier: 1.0,
    uiSignal: 'WARN',
    interruptionToleranceMs: 400,
  },
  confident: {
    systemPromptInjection: "NOTICE: The student is highly engaged and confident. Validate their accurate thinking briefly, skip basic foundational overviews, and ramp up the technical complexity or introduce edge cases.",
    voicePitchModifier: 1.05, // Energetic, slightly faster validation response
    uiSignal: 'CELEBRATE',
    interruptionToleranceMs: 600, // Firm execution to finish presenting the next challenging problem
  },
  neutral: {
    systemPromptInjection: "Maintain standard rigorous technical AI interviewer pacing. Continue executing the scheduled programming benchmark curriculum.",
    voicePitchModifier: 1.0,
    uiSignal: 'NORMAL',
    interruptionToleranceMs: 500,
  }
};
