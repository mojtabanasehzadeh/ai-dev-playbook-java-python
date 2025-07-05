package com.moji.ai.dto;

// ConsensusPrediction.java - Create this as a separate file or add to your controller file
public record ConsensusPrediction(
        int finalPrediction,           // 0 = DELAYED, 1 = ON_TIME
        double consensusConfidence,    // How confident we are (0.0 to 1.0)
        String message,               // Human-readable result
        PredictionResponse pythonResult,  // What Python model said
        JavaMLResult javaResult,      // What Java model said
        String reasoning              // Why we made this decision
) {
}